from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from torch import optim
from torch.utils.data import DataLoader
import argparse
import copy
import logging
import numpy as np
import os, time, csv
import pickle
import torch
import torch.nn.functional as F
from datetime import datetime
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from settings import parse_test_args, model_classes, init_logging
from utils import TextClassificationDataset, dynamic_collate_fn, prepare_inputs, DynamicBatchSampler


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


def test_task(task_id, args, model, test_dataset):

    if not args.no_fp16_test:
        model = model.half()

    def update_metrics(loss, logits, cur_loss, cur_acc):
        preds = np.argmax(logits, axis=1)
        return cur_loss + loss, cur_acc + np.sum(preds == labels.detach().cpu().numpy())

    cur_loss, cur_acc = 0, 0
    all_labels, all_label_confs, all_preds = [], [], []
    if args.adapt_steps >= 1:
        with torch.no_grad():
            org_params = torch.cat([torch.reshape(param, [-1]) for param in model.parameters()], 0)

        q_input_ids = pickle.load(open(os.path.join(args.output_dir, 'q_input_ids-{}'.format(task_id)), 'rb'))
        q_masks = pickle.load(open(os.path.join(args.output_dir, 'q_masks-{}'.format(task_id)), 'rb'))
        q_labels = pickle.load(open(os.path.join(args.output_dir, 'q_labels-{}'.format(task_id)), 'rb'))

        for i in range(len(test_dataset)):
            labels, input_ids = test_dataset[i]
            labels = torch.tensor(np.expand_dims(labels, 0), dtype=torch.long).cuda()
            input_ids = torch.tensor(np.expand_dims(input_ids, 0), dtype=torch.long).cuda()
            loss, logits, softmax = local_adapt(input_ids, labels, copy.deepcopy(model), q_input_ids[i], q_masks[i], q_labels[i], args, org_params)
            # Output has size of torch.Size([16, 33]) [BATCH, CLASSES]
            label_conf = softmax[np.arange(len(softmax)), labels] # Select labels in the softmax of 33 classes
            preds = np.argmax(logits, axis=1)
            
            cur_loss, cur_acc = update_metrics(loss, logits, cur_loss, cur_acc)
            
            # Append all!
            all_labels.extend(labels.tolist())
            all_label_confs.extend(label_conf.tolist())
            all_preds.extend(preds.tolist())
            
            if (i+1) % args.logging_steps == 0:
                logging.info("Local adapted {}/{} examples, test loss: {:.3f} , test acc: {:.3f}".format(
                    i+1, len(test_dataset), cur_loss/(i+1), cur_acc/(i+1)))
    else:
        test_dataloader = DataLoader(test_dataset, num_workers=args.n_workers, collate_fn=dynamic_collate_fn,
                                     batch_sampler=DynamicBatchSampler(test_dataset, args.batch_size * 4))
        tot_n_inputs = 0
        for step, batch in enumerate(test_dataloader):
            n_inputs, input_ids, masks, labels = prepare_inputs(batch)
            tot_n_inputs += n_inputs
            with torch.no_grad():
                model.eval()
                outputs = model(input_ids=input_ids, attention_mask=masks, labels=labels)
                loss = outputs[0].item()
                logits = outputs[1].detach().cpu().numpy()
            cur_loss, cur_acc = update_metrics(loss*n_inputs, logits, cur_loss, cur_acc)
            if (step+1) % args.logging_steps == 0:
                logging.info("Tested {}/{} examples , test loss: {:.3f} , test acc: {:.3f}".format(
                    tot_n_inputs, len(test_dataset), cur_loss/tot_n_inputs, cur_acc/tot_n_inputs))
        assert tot_n_inputs == len(test_dataset)


    logger.info("test loss: {:.3f} , test acc: {:.3f}".format(
        cur_loss / len(test_dataset), cur_acc / len(test_dataset)))
    return cur_acc / len(test_dataset), all_labels, all_label_confs, all_preds


def main():
    args = parse_test_args()
    train_args = pickle.load(open(os.path.join(args.output_dir, 'train_args'), 'rb'))
    assert train_args.output_dir == args.output_dir
    args.__dict__.update(train_args.__dict__)
    init_logging(os.path.join(args.output_dir, 'log_test.txt'))
    logger.info("args: " + str(args))

    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    save_model_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(len(args.tasks)-1))
    model = model_class.from_pretrained(save_model_path, config=model_config).cuda()

    avg_acc = 0
    accuracies = []
    data_for_visual = []
    for task_id, task in enumerate(args.tasks):
        logger.info("Start testing {}...".format(task))
        test_dataset = pickle.load(open(os.path.join(args.output_dir, 'test_dataset-{}'.format(task_id)), 'rb'))
        task_acc, all_labels, all_label_confs, all_preds = test_task(task_id, args, model, test_dataset)
        
        # Start Edit
        data_ids = [task + str(i) for i in range(len(all_labels))]
        data_for_visual.extend(list(zip(data_ids, all_labels, all_label_confs, all_preds)))
        accuracies.append(task_acc)
        
        avg_acc += task_acc / len(args.tasks)
        
    # Start Edit
    print()
    print("COPY PASTA - not really but ok")
    for row in accuracies:
        print(row)
    print()
    print('Overall test metrics:')
    logger.info("Average acc: {:.4f}".format(avg_acc))
    
    _model_path0 = os.path.splitext(save_model_path)[0]
    csv_filename = _model_path0 + "_update"+ str(args.adapt_steps) +"_results.csv" # for selective replay
    with open(csv_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["data_idx", "label", "label_conf", "pred"])
        csv_writer.writerows(data_for_visual)
    print(f"Done writing CSV File at {csv_filename}")


if __name__ == "__main__":
    logger.info(f"[TIME] Start Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    main()
    toc_RUN = time.time() - tic_RUN
    logger.info(f"[TIME] End Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
