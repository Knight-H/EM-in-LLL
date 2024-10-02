from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
from torch import optim
import logging
import numpy as np
import os, time, math, gc
import pickle
import torch
import torchtext
import higher
from tqdm import tqdm
from datetime import datetime
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from memory_rel_v2 import MemoryFewRel
from settings import parse_train_args, model_classes, init_logging
from utils import TextClassificationDataset, DynamicBatchSampler
from utils import dynamic_collate_fn, prepare_inputs, read_relations, read_rel_data, \
                    get_relation_embedding, prepare_rel_datasets, rel_encode, replicate_rel_data, \
                    get_relation_index, create_relation_clusters, LifelongFewRelDataset, calculate_accuracy, make_rel_prediction
from transformers import BertTokenizer # Use the new BertTokenizer for batch_encode_plus!

loss_fn = torch.nn.BCEWithLogitsLoss()

# Training v3 - use Local Adaptation just like how the Meta-MbPA testing works
#    For FewRel ! 
def encode_text(tokenizer, text):
    encode_result = tokenizer.batch_encode_plus(text, return_token_type_ids=False, padding='longest', return_tensors='pt')
    for key in encode_result:
        encode_result[key] = encode_result[key].cuda()
    return len(encode_result["input_ids"]), encode_result["input_ids"], encode_result["attention_mask"]

def train_task(args, model, memory, train_datasets, tokenizer):
    for train_idx, train_dataset in enumerate(train_datasets):
        logger.info('Starting with train_idx: {}'.format(train_idx))
        # Change to each dataset.
        train_dataloader = DataLoader(train_dataset, num_workers=args.n_workers, batch_size=args.batch_size, shuffle=False, collate_fn=rel_encode)

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
    
        pbar = tqdm(total=math.ceil(len(train_dataset) / args.batch_size))
        for step, batch in enumerate(train_dataloader):
            model.train()
            text, labels, candidates = batch
            replicated_text, replicated_relations, ranking_label = replicate_rel_data(text,labels,candidates)
            n_inputs, input_ids, masks = encode_text(tokenizer, list(zip(replicated_text, replicated_relations)))
            input_ids_select = torch.tensor(ranking_label) == 1 # True/ false for selecting adding in memory [1,0,0,1,0,0...] --> [T,F,F, T,F,F... ]
            targets = torch.tensor(ranking_label).float().unsqueeze(1).cuda()
            pbar.update(1)    

            # ----- Knight Added -----
            # >> If there is no keys (at least 32), add the first batch first before continuing!
            if len(memory.keys) < 32:
                memory.add(text, labels, candidates, input_ids[input_ids_select], masks[input_ids_select])
                continue
            # 1. Query Neighbors 
            # https://stackoverflow.com/questions/49841324/what-does-calling-fit-multiple-times-on-the-same-model-do
            # there's actually partial_fit(), but I think for now just overwrite
            memory.tree.fit(np.array(memory.keys)) # Fit on current memory
            with torch.no_grad():
                q_text, q_labels, q_candidates = memory.query(input_ids, masks)
                # Get only one is enough! (Batch of 32)
                # Note: Need to find way to improve later
                q_text = q_text[0]
                q_labels = q_labels[0]
                q_candidates = q_candidates[0]

                q_replicated_text, q_replicated_relations, q_ranking_label = replicate_rel_data(q_text,q_labels,q_candidates)
                
#                 print("q_text", q_text)
#                 print("q_text", len(q_text))
#                 print("q_replicated_text", q_replicated_text)
#                 print("q_replicated_text", len(q_replicated_text))
#                 print("q_replicated_relations", q_replicated_relations)
#                 print("q_replicated_relations", len(q_replicated_relations))
                
                
#                 print("q_text", q_text)
#                 print("q_text", len(q_text))
#                 print("q_labels", q_labels)
#                 print("q_labels", len(q_labels))
                
#                 print("q_candidates", q_candidates)
#                 print("q_candidates", len(q_candidates))
                q_n_inputs, q_input_ids, q_masks = encode_text(tokenizer, list(zip(q_replicated_text, q_replicated_relations)))
                q_targets = torch.tensor(q_ranking_label).float().unsqueeze(1).cuda()
                    
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
                    output = fmodel(input_ids=q_input_ids, attention_mask=q_masks)[0]
                    loss = loss_fn(output, q_targets) + args.adapt_lambda * torch.sum((org_params - params)**2)  
                    diffopt.step(loss)
                    fmodel.zero_grad() # Is this necessary? but local adapt have this!
                # ------- ENDED -----------
                # 2.2 Outer Loop (Query Set)
                memory.add(text, labels, candidates, input_ids[input_ids_select], masks[input_ids_select], args.write_prob)
                output = fmodel(input_ids=input_ids, attention_mask=masks)[0]
                loss = loss_fn(output, targets)
                
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
                tot_n_inputs += n_inputs # <--  n_inputs is amplified in fewrel!!
                tot_epoch_loss += loss.item() * n_inputs

                if (step+1) % args.logging_steps == 0:
                    logger.info("progress: {:.2f} , step: {} , lr: {:.2E} , avg batch size: {:.1f} , avg loss: {:.3f}".format(
                        tot_n_inputs/args.n_train, step+1, scheduler.get_last_lr()[0], tot_n_inputs//(step+1), tot_epoch_loss/tot_n_inputs))
            
            # 3. Check for replay
            if args.replay_interval >= 1 and (step+1) % args.replay_interval == 0:
                torch.cuda.empty_cache()
                del loss, input_ids, masks, labels
                # 3.0 Prepare Samples, Get real sample from memory & get KNN
                text, labels, candidates = memory.sample(len(text)) # This is much better as batch size, since -> tot_n_inputs // (step + 1)  will diverge as tot_n_inputs go up
                replicated_text, replicated_relations, ranking_label = replicate_rel_data(text,labels,candidates)
                n_inputs, input_ids, masks = encode_text(tokenizer, list(zip(replicated_text, replicated_relations)))
                targets = torch.tensor(ranking_label).float().unsqueeze(1).cuda()

                memory.tree.fit(np.array(memory.keys)) # Fit on current memory
                with torch.no_grad():
                    q_text, q_labels, q_candidates = memory.query(input_ids, masks)
                    # Get only one is enough! (Batch of 32)
                    # Note: Need to find way to improve later
                    q_text = q_text[0]
                    q_labels = q_labels[0]
                    q_candidates = q_candidates[0]

                    q_replicated_text, q_replicated_relations, q_ranking_label = replicate_rel_data(q_text,q_labels,q_candidates)
                    q_n_inputs, q_input_ids, q_masks = encode_text(tokenizer, list(zip(q_replicated_text, q_replicated_relations)))
                    q_targets = torch.tensor(q_ranking_label).float().unsqueeze(1).cuda()
                    
                # Debugging!
                logger.info(f"I am replaying! on step {step+1} with memory of size {len(memory.keys)} and sampling size {tot_n_inputs // (step + 1)}")
                #print(f"This is what I Sample {input_ids}")
                #print(f"This is what I Support  {q_input_ids}")
                # Meta Learning
                inner_optimizer.zero_grad()
                # Get Original Params
                with torch.no_grad():
                    org_params = torch.cat([torch.reshape(param, [-1]) for param in model.parameters()], 0)

                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False,track_higher_grads=False) as (fmodel, diffopt):
                    # 3.1 Inner Loop (Support Set)
                    for _step in range(args.adapt_steps):
                        params = torch.cat([torch.reshape(param, [-1]) for param in fmodel.parameters()], 0)
                        output = fmodel(input_ids=q_input_ids, attention_mask=q_masks)[0]
                        loss = loss_fn(output, q_targets) + args.adapt_lambda * torch.sum((org_params - params)**2)  
                        diffopt.step(loss)
                    # 3.2 Outer Loop (Query Set)
                    output = fmodel(input_ids=input_ids, attention_mask=masks)[0]
                    loss = loss_fn(output, targets)
                    logger.info(f"Replay loss {loss.item()} with tot_n_inputs {tot_n_inputs} and n_inputs from train {n_inputs}")
                    
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
                    

        pbar.close()
        logger.info("Finish training, avg loss: {:.3f}".format(tot_epoch_loss/tot_n_inputs))
        del meta_optimizer, optimizer_grouped_parameters
        #assert tot_n_inputs == len(train_dataset) == args.n_train
        logger.info(f"tot_n_inputs: {tot_n_inputs}")
        logger.info(f"len(train_dataset): {len(train_dataset)}")
        logger.info(f"args.n_train: {args.n_train}")


def evaluate(args, model, memory, test_dataloader, tokenizer):
    # Before anything else, just sample randomly from memory!
    # Use this as sample going forward!
    text, labels, candidates = memory.sample(32)
    replicated_text, replicated_relations, ranking_label = replicate_rel_data(text,labels,candidates)
    n_inputs, input_ids, masks = encode_text(tokenizer, list(zip(replicated_text, replicated_relations)))
    targets = torch.tensor(ranking_label).float().unsqueeze(1).cuda()
    print(f"Total No.# of sampled: {len(labels)}")
    # query like this first just like training... this will need to be removed later!!! so we can adapt 32x32
    memory.tree.fit(np.array(memory.keys)) # Fit on current memory
    with torch.no_grad():
        q_text, q_labels, q_candidates = memory.query(input_ids, masks)
        # Get only one is enough! (Batch of 32)
        # Note: Need to find way to improve later
        q_text = q_text[0]
        q_labels = q_labels[0]
        q_candidates = q_candidates[0]

        q_replicated_text, q_replicated_relations, q_ranking_label = replicate_rel_data(q_text,q_labels,q_candidates)
        q_n_inputs, q_input_ids, q_masks = encode_text(tokenizer, list(zip(q_replicated_text, q_replicated_relations)))
        q_targets = torch.tensor(q_ranking_label).float().unsqueeze(1).cuda()
    
    # Meta-Learning Local Adaptation
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    inner_optimizer = optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Get Original Params
    with torch.no_grad():
        org_params = torch.cat([torch.reshape(param, [-1]) for param in model.parameters()], 0)

    all_losses, all_predictions, all_labels = [], [], []
    with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False,track_higher_grads=False) as (fmodel, diffopt):
        # 1 Inner Loop (Support Set) - Once for all testing samples
        for step in range(args.adapt_steps):
            params = torch.cat([torch.reshape(param, [-1]) for param in fmodel.parameters()], 0)
            output = fmodel(input_ids=q_input_ids, attention_mask=q_masks)[0]
            loss = loss_fn(output, q_targets) + args.adapt_lambda * torch.sum((org_params - params)**2)  
            diffopt.step(loss)
            fmodel.zero_grad() # Is this necessary? but local adapt have this!
            
        # 2 Outer Loop (Query Set)
        tot_n_inputs = 0
        for step, batch in enumerate(test_dataloader):
            text, labels, candidates = batch
            replicated_text, replicated_relations, ranking_label = replicate_rel_data(text,labels,candidates)
            n_inputs, input_ids, masks = encode_text(tokenizer, list(zip(replicated_text, replicated_relations)))
            targets = torch.tensor(ranking_label).float().unsqueeze(1).cuda()
            tot_n_inputs += n_inputs
            
            with torch.no_grad():
                fmodel.eval()
                output = fmodel(input_ids=input_ids, attention_mask=masks)[0]
                loss = loss_fn(output, targets)
            loss = loss.item()
            pred, true_labels = make_rel_prediction(output, ranking_label)

            # Append all!
            all_losses.append(loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(true_labels.tolist())

    acc = calculate_accuracy(all_predictions, all_labels)
    logger.info("Test Metrics: Loss = {:.4f} , test acc: {:.3f}".format(np.mean(all_losses), acc))
    return acc


def test_task(args, model, memory, test_dataset, tokenizer):
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=rel_encode)
    acc = evaluate(args, model, memory, test_dataloader, tokenizer)
    logger.info('Overall test metrics: Accuracy = {:.4f}'.format(acc))
    return acc

def main():
    args = parse_train_args()

    # Load training and validation data
    logger.info('Loading the dataset')
    data_dir = '/data/omler_data/LifelongFewRel'
    relation_file = os.path.join(data_dir, 'relation_name.txt')
    training_file = os.path.join(data_dir, 'training_data.txt')
    validation_file = os.path.join(data_dir, 'val_data.txt')
    # ie. ['fill', ['place', 'served', 'by', 'transport', 'hub'], ['mountain', 'range'], ['religion'], ['participating', 'team'], ...]
    # Note that 'fill' is the 0 index, can be ignored
    relation_names = read_relations(relation_file) # List of relation names (converted to 1-based index later)
    train_data = read_rel_data(training_file)
    val_data = read_rel_data(validation_file)
    logger.info('Finished loading the dataset')
    # label2idx is reverse of relation_names,  where we map label (space joined relation) --> label id. ie. "work location" --> 2
    # Used for Label Aware ER -- maybe not need if we just use list as key???
    label2idx = {" ".join(relation_name): i for i,relation_name in enumerate(relation_names) if i != 0}
    args.idx2label = relation_names
    args.label2idx = label2idx

    # Load GloVe vectors
    logger.info('Loading GloVe vectors')
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    logger.info('Finished loading GloVe vectors')

    # Get relation embeddings for clustering
    relation_embeddings = get_relation_embedding(relation_names, glove)
    print(relation_embeddings.shape)

    # Generate clusters
    # This essentially goes through all train_data and get label set, which is a list of 1-80 ie. [80, 25, 75, 15, 62, 74, 5, 10...] 
    relation_index = get_relation_index(train_data)  
    # This uses KMeans to divide the label up into 10 disjoint clusters ie. {80: 1, 25: 5, 75: 3, 15: 1, 62: 1, 74: 1, 5: 1, 10: 2...}
    # > relation_embeddings just return a dictionary of relation_index --> Glove embedding ie. { 80: embedding, 25: embedding, ...}
    cluster_labels, relation_embeddings = create_relation_clusters(args.num_clusters, relation_embeddings, relation_index)

    # Validation Datast v2 for Task-Aware , Separate it into the 3 clusters
    # I forgot why we did this, but do it first. 
#     val_dataset = [LifelongFewRelDataset(vd, relation_names) for vd in split_rel_data_by_clusters(val_data, cluster_labels, args.num_clusters, list(range(args.num_clusters)))]
#     print(f"Val Dataset2 Length: {[len(x) for x in val_dataset]}")
#     print(f"Val Dataset2 Sum: {sum([len(x) for x in val_dataset])}")
    # Validation dataset (Test Dataset)
    val_dataset = LifelongFewRelDataset(val_data, relation_names)
    print(f"Val Dataset Length: {len(val_dataset)}")
    
    # Run for different orders of the clusters
    accuracies = []
    for i in range(args.order):
        logger.info('Running order {}'.format(i + 1))

        # Initialize Training args and Model, Tokenizer
        pickle.dump(args, open(os.path.join(args.output_dir, 'train_args'), 'wb'))
        init_logging(os.path.join(args.output_dir, 'log_train.txt'))
        logger.info("args: " + str(args))

        logger.info("Initializing main {} model".format(args.model_name))
        config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
        #tokenizer = args.tokenizer_class.from_pretrained(args.model_name)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        model_config = config_class.from_pretrained(args.model_name, num_labels=1)
        config_save_path = os.path.join(args.output_dir, 'config')
        model_config.to_json_file(config_save_path)
        model = model_class.from_pretrained(args.model_name, config=model_config).cuda()
        memory = MemoryFewRel(args)

        logger.info('Using {} as learner'.format(model.__class__.__name__))

        # Generate continual learning training data
        logger.info('Generating continual learning data')
        train_datasets, shuffle_index = prepare_rel_datasets(train_data, relation_names, cluster_labels, args.num_clusters)
        args.shuffle_index = shuffle_index
        logger.info(f"Shuffle Index: {shuffle_index}")
        logger.info(f"Train Dataset Length: {[len(x) for x in train_datasets]}")
        logger.info('Finished generating continual learning data')

        # Training
        logger.info('----------Training starts here----------')
        tic_TRAIN = time.time()
        train_task(args, model, memory, train_datasets, tokenizer)
        model_save_path = os.path.join(args.output_dir, 'checkpoint-order{}'.format(i))
        torch.save(model.state_dict(), model_save_path)
        logger.info('Saved the model with name {}'.format(model_save_path))
        pickle.dump(memory, open(os.path.join(args.output_dir, 'memory-order{}'.format(i)), 'wb'))
        toc_TRAIN = time.time() - tic_TRAIN
        logger.info(f"[TIME] Training Time within {toc_TRAIN/3600} hours")

        # Testing
        logger.info('----------Testing starts here----------')
        tic_TEST = time.time()
        acc = test_task(args, model, memory, val_dataset, tokenizer)
        accuracies.append(acc)
        toc_TEST = time.time() - tic_TEST
        logger.info(f"[TIME] Testing Time within {toc_TEST/3600} hours")

        # Delete the model to free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info('Accuracy across runs = {}'.format(accuracies))
    logger.info('Average accuracy across runs: {}'.format(np.mean(accuracies)))

if __name__ == "__main__":
    
    logger.info(f"[TIME] Start Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    main()
    toc_RUN = time.time() - tic_RUN
    logger.info(f"[TIME] End Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
