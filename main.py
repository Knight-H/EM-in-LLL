from apex import amp
import os
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import copy
import logging
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from settings import parse_args, model_classes
from utils import TextClassificationDataset, dynamic_collate_fn, prepare_inputs, TimeFilter
from memory import Memory


def local_adapt(input_ids, labels, q_input_ids, q_masks, q_labels, tmp_model, args, org_params):
    q_input_ids = q_input_ids.to(args.device).detach()
    q_masks = q_masks.to(args.device).detach()
    q_labels = q_labels.to(args.device).detach()

    optimizer = optim.SGD(tmp_model.parameters(), lr=args.local_adapt_lr, momentum=0.9)
    if args.fp16_test:
        tmp_model, optimizer = amp.initialize(tmp_model, optimizer, opt_level="O3", verbosity=0)

    tmp_model.zero_grad()
    for step in range(args.adapt_steps):
        tmp_model.train()
        loss = tmp_model(input_ids=q_input_ids, attention_mask=q_masks, labels=q_labels)[0] \
            + args.local_lambda * torch.sum((org_params - torch.cat([param.data.view(-1) for param in tmp_model.parameters()], 0))**2)
        if args.fp16_test:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        tmp_model.zero_grad()

    with torch.no_grad():
        tmp_model.eval()
        output = tmp_model(input_ids=input_ids, labels=labels)[:2]
        del tmp_model, optimizer, input_ids, labels, q_input_ids, q_masks, q_labels
        torch.cuda.empty_cache()
        return output


def test_task(args, model, memory, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 6,
                                 num_workers=args.n_workers, collate_fn=dynamic_collate_fn)

    def update_metrics(loss, logits, cur_loss, cur_acc):
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        return cur_loss + loss, cur_acc + np.sum(preds == labels.detach().cpu().numpy())

    cur_loss, cur_acc = 0, 0
    if args.adapt_steps >= 1:
        with torch.no_grad():
            model_config = args.config_class.from_pretrained(args.model_name, num_labels=args.n_labels, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
            save_model_path = os.path.join(args.output_dir, 'checkpoint')
            org_model = args.model_class.from_pretrained(save_model_path, config=model_config)
            org_model.to(args.device)
            org_params = torch.cat([param.data.view(-1) for param in org_model.parameters()], 0)
            if args.fp16_test:
                org_params = org_params.half()
            del org_model
    
        q_input_ids, q_masks, q_labels = [], [], []
        for step, batch in enumerate(test_dataloader):
            n_inputs, input_ids, masks, labels = prepare_inputs(batch, args.device)
            with torch.no_grad():
                cur_q_input_ids, cur_q_masks, cur_q_labels = memory.query(input_ids, masks)
            q_input_ids.extend(cur_q_input_ids)
            q_masks.extend(cur_q_masks)
            q_labels.extend(cur_q_labels)

        for i in range(len(test_dataset)):
            labels, input_ids = test_dataset[i]
            labels = torch.tensor(np.expand_dims(labels, 0), dtype=torch.long).to(args.device)
            input_ids = torch.tensor(np.expand_dims(input_ids, 0), dtype=torch.long).to(args.device)
            loss, logits = local_adapt(input_ids, labels, q_input_ids[i], q_masks[i], q_labels[i], copy.deepcopy(model), args, org_params)
            cur_loss, cur_acc = update_metrics(loss.item(), logits, cur_loss, cur_acc)
            if (i+1) % args.logging_steps == 0:
                logging.info("Local adapted {}/{} examples, test loss: {} , test acc: {}".format(
                    i+1, len(test_dataset), cur_loss/(i+1), cur_acc/(i+1)))
    else:
        tot_n_inputs = 0
        for step, batch in enumerate(test_dataloader):
            n_inputs, input_ids, masks, labels = prepare_inputs(batch, args.device)
            tot_n_inputs += n_inputs
            with torch.no_grad():
                model.eval()
                outputs = model(input_ids=input_ids, attention_mask=masks, labels=labels)
                loss, logits = outputs[:2]
            cur_loss, cur_acc = update_metrics(loss.item()*n_inputs, logits, cur_loss, cur_acc)
            if (step+1) % args.logging_steps == 0:
                logging.info("Tested {}/{} examples , test loss: {} , test acc: {}".format(
                    tot_n_inputs, len(test_dataset), cur_loss/tot_n_inputs, cur_acc/tot_n_inputs))


    logger.info("test loss: {:.3f} , test acc: {:.3f}".format(
        cur_loss / len(test_dataset), cur_acc / len(test_dataset)))
    return cur_acc / len(test_dataset)


def train_task(args, model, memory, train_dataset, valid_dataset):

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=not args.reproduce,
                                  num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    if valid_dataset:
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size * 6,
                                      num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    tot_train_step = len(train_dataloader)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=tot_train_step)

    updates_per_epoch = len(train_dataset)
    global_step = 0
    model.zero_grad()
    tot_epoch_loss, tot_n_inputs = 0, 0

    def update_parameters(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scheduler.step()
        optimizer.step()
        model.zero_grad()

    for step, batch in enumerate(train_dataloader):
        model.train()
        n_inputs, input_ids, masks, labels = prepare_inputs(batch, args.device)
        memory.add(input_ids, masks, labels)
        loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
        update_parameters(loss)
        global_step += 1
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        if global_step % args.logging_steps == 0:
            logger.info("progress: {:.2f} , global step: {} , lr: {:.2E} , avg loss: {:.3f}".format(
                tot_n_inputs/updates_per_epoch, global_step, scheduler.get_lr()[0], tot_epoch_loss/tot_n_inputs))

        if args.replay_interval >= 1 and (step+1) % args.replay_interval == 0:
            input_ids, masks, labels = memory.sample(args.batch_size)
            loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
            update_parameters(loss)

        del loss, input_ids, masks, labels

    logger.info("Finsih training, global step: {} , avg loss: {:.3f}".format(
        global_step, tot_epoch_loss/tot_n_inputs))
    del optimizer, optimizer_grouped_parameters
    assert tot_n_inputs == len(train_dataset)


def main():
    args = parse_args()


    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format,
                        filename=os.path.join(args.output_dir, 'log.txt'),
                        filemode='w', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

    logger.info("args: " + str(args))

    args.config_class, args.model_class, args.tokenizer_class = model_classes[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name)

    model_config = args.config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    config_save_path = os.path.join(args.output_dir, 'config')
    model_config.to_json_file(config_save_path)
    logger.info("Loading main {} model".format(args.model_name))
    model = args.model_class.from_pretrained(args.model_name, config=model_config)
    model.to(args.device)
    memory = Memory(args)

    for task in args.tasks:
        train_dataset = TextClassificationDataset(task, "train", args, tokenizer)

        if args.valid_ratio > 0:
            valid_dataset = TextClassificationDataset(task, "valid", args, tokenizer)
        else:
            valid_dataset = None

        logger.info("Start training {}...".format(task))
        train_task(args, model, memory, train_dataset, valid_dataset)
        model_save_path = os.path.join(args.output_dir, 'checkpoint')
        torch.save(model.state_dict(), model_save_path)
        torch.cuda.empty_cache()

    if args.adapt_steps >= 1:
        memory.build_tree()

    avg_acc = 0
    for task in args.tasks:
        test_dataset = TextClassificationDataset(task, "test", args, tokenizer)
        logger.info("Start testing {}...".format(task))
        task_acc = test_task(args, model, memory, test_dataset)
        avg_acc += task_acc / len(args.tasks)
    logger.info("Average acc: {:.3f}".format(avg_acc))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark=True
    main()
