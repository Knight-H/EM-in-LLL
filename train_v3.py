from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
from torch import optim
import logging
import numpy as np
import os, time
import pickle
import torch
import higher
from tqdm import tqdm
from datetime import datetime
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from memory_v2 import Memory
from settings import parse_train_args, model_classes, init_logging
from utils import TextClassificationDataset, DynamicBatchSampler
from utils import dynamic_collate_fn, prepare_inputs


# Training v3 - use Local Adaptation just like how the Meta-MbPA testing works


def local_adapt(input_ids, labels, tmp_model, q_input_ids, q_masks, q_labels, args, org_params):
    q_input_ids = q_input_ids.cuda().detach()
    q_masks = q_masks.cuda().detach()
    q_labels = q_labels.cuda().detach()

    optimizer = optim.SGD(tmp_model.parameters(), lr=args.adapt_lr, momentum=0.9)

    tmp_model.zero_grad()
    for step in range(args.adapt_steps):
        tmp_model.train()
        params = torch.cat([torch.reshape(param, [-1]) for param in tmp_model.parameters()], 0)
        loss = tmp_model(input_ids=q_input_ids, attention_mask=q_masks, labels=q_labels)[0] \
            + args.adapt_lambda * torch.sum((org_params - params)**2)
        loss.backward()
        optimizer.step()
        tmp_model.zero_grad()

    with torch.no_grad():
        tmp_model.eval()
        output = tmp_model(input_ids=input_ids, labels=labels)[:2]
        loss = output[0].item()
        logits = output[1].detach().cpu().numpy()
        softmax = F.softmax(output[1], -1) # Added this before empty cache? for logits/output[1]!
        torch.cuda.empty_cache()
        return loss, logits, softmax


def train_task(args, model, memory, train_dataset, valid_dataset=None):

    train_dataloader = DataLoader(train_dataset, num_workers=args.n_workers, collate_fn=dynamic_collate_fn,
                                  batch_sampler=DynamicBatchSampler(train_dataset, args.batch_size))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    inner_optimizer = optim.SGD(optimizer_grouped_parameters, lr=args.inner_lr, momentum=0.9)
    meta_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(meta_optimizer, warmup_steps=args.warmup_steps, t_total=len(train_dataset)//10)

    model.zero_grad()
    tot_epoch_loss, tot_n_inputs = 0, 0
    
    pbar = tqdm(total=len(train_dataset))
    for step, batch in enumerate(train_dataloader):
        model.train()
        n_inputs, input_ids, masks, labels = prepare_inputs(batch)
        pbar.update(n_inputs)

        # ----- Knight Added -----
        # >> If there is no keys (at least 32), add the first batch first before continuing!
        if len(memory.keys) < 32:
            memory.add(input_ids, masks, labels)
            continue
        # 1. Query Neighbors 
        # https://stackoverflow.com/questions/49841324/what-does-calling-fit-multiple-times-on-the-same-model-do
        # there's actually partial_fit(), but I think for now just overwrite
        memory.tree.fit(np.array(memory.keys)) # Fit on current memory
        with torch.no_grad():
            q_input_ids, q_masks, q_labels = memory.query(input_ids, masks)
            # Get only one is enough! (Batch of 32)
            # Note: Need to find way to improve later
            q_input_ids = q_input_ids[0].cuda()
            q_masks     = q_masks[0].cuda()
            q_labels    = q_labels[0].cuda()

        # 2. Meta Learning
        inner_optimizer.zero_grad()
        # Get Original Params
        with torch.no_grad():
            org_params = torch.cat([torch.reshape(param, [-1]) for param in model.parameters()], 0)

        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False,track_higher_grads=False) as (fmodel, diffopt):
            # 2.1 Inner Loop (Support Set)
            # q_input_ids will be a kust if 32 tensors of length 1 now.
            for _step in range(args.adapt_steps):
                params = torch.cat([torch.reshape(param, [-1]) for param in fmodel.parameters()], 0)
                loss = fmodel(input_ids=q_input_ids, attention_mask=q_masks, labels=q_labels)[0]\
                        + args.adapt_lambda * torch.sum((org_params - params)**2)
                diffopt.step(loss)
                fmodel.zero_grad() # Is this necessary? but local adapt have this!
            # ------- ENDED -----------
            # 2.2 Outer Loop (Query Set)
            memory.add(input_ids, masks, labels, args.write_prob)
            loss = fmodel(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
            
            # meta gradients - copied from # PLN meta gradients
            _params = [p for p in fmodel.parameters() if p.requires_grad]
            meta_grads = torch.autograd.grad(loss, _params)
            _params = [p for p in model.parameters() if p.requires_grad]
            for param, meta_grad in zip(_params, meta_grads):
                if param.grad is not None:
                    param.grad += meta_grad.detach()
                else:
                    param.grad = meta_grad.detach()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            meta_optimizer.step()
            meta_optimizer.zero_grad()
            scheduler.step()
            tot_n_inputs += n_inputs
            tot_epoch_loss += loss.item() * n_inputs

            if (step+1) % args.logging_steps == 0:
                logger.info("progress: {:.2f} , step: {} , lr: {:.2E} , avg batch size: {:.1f} , avg loss: {:.3f}".format(
                    tot_n_inputs/args.n_train, step+1, scheduler.get_last_lr()[0], tot_n_inputs//(step+1), tot_epoch_loss/tot_n_inputs))
        
        # 3. Check for replay
        if args.replay_interval >= 1 and (step+1) % args.replay_interval == 0:
            torch.cuda.empty_cache()
            del loss, input_ids, masks, labels
            # 3.0 Prepare Samples, Get real sample from memory & get KNN
            input_ids, masks, labels = memory.sample(tot_n_inputs // (step + 1))
            memory.tree.fit(np.array(memory.keys)) # Fit on current memory
            with torch.no_grad():
                q_input_ids, q_masks, q_labels = memory.query(input_ids, masks)
                
                # Get only one is enough! (Batch of 32)
                # Note: Need to find way to improve later
                q_input_ids = q_input_ids[0].cuda()
                q_masks     = q_masks[0].cuda()
                q_labels    = q_labels[0].cuda()
                
            # Debugging!
            print(f"I am replaying! on step {step+1} with memory of size {len(memory.keys)} and sampling size {tot_n_inputs // (step + 1)}")
            #print(f"This is what I Sample {input_ids}")
            #print(f"This is what I Support  {q_input_ids}")
            # Meta Learning
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False,track_higher_grads=False) as (fmodel, diffopt):
                # 3.1 Inner Loop (Support Set)
                for _step in range(args.adapt_steps):
                    params = torch.cat([torch.reshape(param, [-1]) for param in fmodel.parameters()], 0)
                    loss = fmodel(input_ids=q_input_ids, attention_mask=q_masks, labels=q_labels)[0]\
                        + args.adapt_lambda * torch.sum((org_params - params)**2)
                    diffopt.step(loss)
                # 3.2 Outer Loop (Query Set)
                loss = fmodel(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
                print(f"Replay loss {loss.item()} with tot_n_inputs {tot_n_inputs} and n_inputs from train {n_inputs}")
                
                # meta gradients - copied from # PLN meta gradients
                _params = [p for p in fmodel.parameters() if p.requires_grad]
                meta_grads = torch.autograd.grad(loss, _params)
                _params = [p for p in model.parameters() if p.requires_grad]
                for param, meta_grad in zip(_params, meta_grads):
                    if param.grad is not None:
                        param.grad += meta_grad.detach()
                    else:
                        param.grad = meta_grad.detach()
                
                torch.nn.utils.clip_grad_norm_(fmodel.parameters(), args.max_grad_norm)
                meta_optimizer.step()
                meta_optimizer.zero_grad()
                scheduler.step()
                

    pbar.close()
    logger.info("Finish training, avg loss: {:.3f}".format(tot_epoch_loss/tot_n_inputs))
    del meta_optimizer, optimizer_grouped_parameters
    #assert tot_n_inputs == len(train_dataset) == args.n_train
    logger.info(f"tot_n_inputs: {tot_n_inputs}")
    logger.info(f"len(train_dataset): {len(train_dataset)}")
    logger.info(f"args.n_train: {args.n_train}")


def main():
    args = parse_train_args()
    pickle.dump(args, open(os.path.join(args.output_dir, 'train_args'), 'wb'))
    init_logging(os.path.join(args.output_dir, 'log_train.txt'))
    logger.info("args: " + str(args))

    logger.info("Initializing main {} model".format(args.model_name))
    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name)

    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    config_save_path = os.path.join(args.output_dir, 'config')
    model_config.to_json_file(config_save_path)
    model = model_class.from_pretrained(args.model_name, config=model_config).cuda()
    memory = Memory(args)

    for task_id, task in enumerate(args.tasks):
        logger.info("Start parsing {} train data...".format(task))
        train_dataset = TextClassificationDataset(task, "train", args, tokenizer)

        logger.info("Start training {}...".format(task))
        train_task(args, model, memory, train_dataset)
        model_save_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(task_id))
        torch.save(model.state_dict(), model_save_path)
        pickle.dump(memory, open(os.path.join(args.output_dir, 'memory-{}'.format(task_id)), 'wb'))

if __name__ == "__main__":
    
    logger.info(f"[TIME] Start Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    main()
    toc_RUN = time.time() - tic_RUN
    logger.info(f"[TIME] End Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
