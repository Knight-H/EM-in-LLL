from multiprocessing import Pool
from torch.utils.data import Dataset, Sampler, random_split
from torch.nn import functional as F
import csv
import numpy as np
import os
import random
import re
import torch
from sklearn.cluster import KMeans
from sklearn import metrics
from settings import label_offsets


def prepare_inputs(batch):
    input_ids, masks, labels = tuple(b.cuda() for b in batch)
    return batch[0].shape[0], input_ids, masks, labels

def pad_to_max_len(input_ids, masks=None):
    max_len = max(len(input_id) for input_id in input_ids)
    masks = torch.tensor([[1]*len(input_id)+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    input_ids = torch.tensor([input_id+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    return input_ids, masks


def dynamic_collate_fn(batch):
    labels, input_ids = list(zip(*batch))
    labels = torch.tensor([b[0] for b in batch], dtype=torch.long)
    input_ids, masks = pad_to_max_len(input_ids)
    return input_ids, masks, labels


class TextClassificationDataset(Dataset):
    def __init__(self, task, mode, args, tokenizer):

        self.task = task
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = tokenizer.max_len
        self.n_test = args.n_test
        self.n_train = args.n_train
        self.valid_ratio = args.valid_ratio

        self.data = []
        self.label_offset = label_offsets[task.split('/')[-1]]
        if self.mode == "test":
            self.fname = os.path.join("/data/omler_data", task, "test.csv")
        elif self.mode in ["train", "valid"]:
            self.fname = os.path.join("/data/omler_data", task, "train.csv")

        with open(self.fname, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                self.data.append(row)

        random.shuffle(self.data)

        if mode == "test":
            self.data = self.data[:self.n_test]
        elif mode == "valid":
            self.data = self.data[:int(self.n_train * self.valid_ratio)]
        elif mode == "train":
            self.data = self.data[int(self.n_train * self.valid_ratio): self.n_train]

        with Pool(args.n_workers) as pool:
            self.data = pool.map(self.map_csv, self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map_csv(self, row):
        context = '[CLS]' + ' '.join(row[1:])[:self.max_len-2] + '[SEP]'
        return (int(row[0]) + self.label_offset, self.tokenizer.encode(context))


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples = len(dataset)

    def __iter__(self):
        max_len = 0
        batch = []
        for idx in np.random.randint(self.n_samples, size=(self.n_samples,), dtype=np.int32):
            if max(max_len, len(self.dataset[idx][1]))**1.17 * (len(batch) + 1) > self.batch_size:
                yield batch
                max_len = 0
                batch = []
            max_len = max(max_len, len(self.dataset[idx][1]))
            batch.append(idx)
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError

######## Added Utils for FewRel
class LifelongFewRelDataset(Dataset):
    def __init__(self, data, relation_names):
        self.relation_names = relation_names
        self.label = []
        self.candidate_relations = []
        self.text = []

        for entry in data:
            self.label.append(self.relation_names[entry[0]])
            negative_relations = entry[1]
            candidate_relation_names = [self.relation_names[x] for x in negative_relations]
            self.candidate_relations.append(candidate_relation_names)
            self.text.append(entry[2])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.text[index], self.label[index], self.candidate_relations[index]

    
def remove_return_sym(str):
    return str.split('\n')[0]
    
def read_relations(relation_file):
    relation_list = ['fill']
    with open(relation_file, encoding='utf8') as file_in:
        for line in file_in:
            line = remove_return_sym(line)
            line = re.sub(r'/', '', line)
            line = line.split()
            relation_list.append(line)
    return relation_list


def read_rel_data(sample_file):
    sample_data = []
    with open(sample_file, encoding='utf8') as file_in:
        for line in file_in:
            items = line.split('\t')
            if len(items[0]) > 0:
                relation_ix = int(items[0])
                if items[1] != 'noNegativeAnswer':
                    candidate_ixs = [int(ix) for ix in items[1].split() if int(ix) != relation_ix]
                    sentence = remove_return_sym(items[2]).split()
                    sample_data.append([relation_ix, candidate_ixs, sentence])
    return sample_data


def get_relation_embedding(relations, glove):
    rel_embed = []
    for rel in relations:
        word_embed = glove.get_vecs_by_tokens(rel, lower_case_backup=True)
        if len(word_embed.shape) == 2:
            rel_embed.append(torch.mean(word_embed, dim=0))
        else:
            rel_embed.append(word_embed)
    rel_embed = torch.stack(rel_embed)
    return rel_embed


def get_relation_index(data):
    relation_pool = []
    for entry in data:
        relation_number = entry[0]
        if relation_number not in relation_pool:
            relation_pool.append(relation_number)
    return relation_pool


def create_relation_clusters(num_clusters, relation_embedding, relation_index):
    ordered_relation_embedding = np.zeros_like(relation_embedding[1:])
    for i, rel_idx in enumerate(relation_index):
        ordered_relation_embedding[i] = relation_embedding[rel_idx]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(ordered_relation_embedding)
    labels = kmeans.labels_
    rel_embed = {}
    cluster_index = {}
    for i in range(len(labels)):
        cluster_index[relation_index[i]] = labels[i]
        rel_embed[relation_index[i]] = relation_embedding[i]
    return cluster_index, rel_embed

# Cluster  Idx  0                        1                      ....  9
#            [ [data0, data1, ...],     [data0, data1, ...] .....   [data0, data1, ...]]
def split_rel_data_by_clusters(data_set, cluster_labels, num_clusters, shuffle_index):
    splitted_data = [[] for i in range(num_clusters)]
    for data in data_set:  # for all 44.5k data
        cluster_number = cluster_labels[data[0]]      # (data[0]) relation idx of that data map--> cluster index
        index_number = shuffle_index[cluster_number]  # cluster index --> order of the shuffle
        splitted_data[index_number].append(data)      # place it in the bucket of num_clusters 
    return splitted_data



def remove_unseen_relations(dataset, seen_relations):
    cleaned_data = []
    for data in dataset:
        neg_cands = [cand for cand in data[1] if cand in seen_relations]
        if len(neg_cands) > 0:
            cleaned_data.append([data[0], neg_cands, data[2]])
        else:
            cleaned_data.append([data[0], data[1][-2:], data[2]])
    return cleaned_data


# Number of clusters = 10
#   shuffle_index       = shuffled num_clusters ( 0, 1, 2, 3 ... 9)
#                         There are 0-9 clusters, the shuffle_index corresponds to which cluster comes first. NOT the cluster idx
#                         ie. [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]  this means
#                         [0 = cluster8, 1 = cluster9, 2= cluster 2,... ]
#   splitted_train_data = Split training dataset by clusters (in shuffled index)
#   For each cluster:
#         add all seen_relations of that cluster
#         current_train_data = remove_unseen_relations in current cluster 
#   train_datasets = [cluster0, cluster1, ... cluster 9]
def prepare_rel_datasets(train_data, relation_names, cluster_labels, num_clusters):
    train_datasets = []

    shuffle_index = list(range(num_clusters))
    random.shuffle(shuffle_index)

    splitted_train_data = split_rel_data_by_clusters(train_data, cluster_labels, num_clusters, shuffle_index)
    seen_relations = []

    for i in range(num_clusters):
        for data_entry in splitted_train_data[i]:
            if data_entry[0] not in seen_relations:
                seen_relations.append(data_entry[0])

        current_train_data = remove_unseen_relations(splitted_train_data[i], seen_relations)

        train_dataset = LifelongFewRelDataset(current_train_data, relation_names)
        train_datasets.append(train_dataset)
    return train_datasets, shuffle_index


def rel_encode(batch):
    text, label, candidate_relations = [], [], []
    for txt, lbl, cand in batch:
        text.append(txt)
        label.append(lbl)
        candidate_relations.append(cand)
    return text, label, candidate_relations

def replicate_rel_data(text, label, candidates):
    replicated_text = []
    replicated_relations = []
    ranking_label = []
    for i in range(len(text)):
        replicated_text.append(text[i])
        replicated_relations.append(label[i])
        ranking_label.append(1)
        for j in range(len(candidates[i])):
            replicated_text.append(text[i])
            replicated_relations.append(candidates[i][j])
            ranking_label.append(0)
    return replicated_text, replicated_relations, ranking_label



def calculate_accuracy(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    return accuracy

def make_rel_prediction(cosine_sim, ranking_label, return_label_conf=False):
    pred, label_confs = [], []
    with torch.no_grad():
        pos_idx = [i for i, lbl in enumerate(ranking_label) if lbl == 1]
        if len(pos_idx) == 1:
            pred.append(torch.argmax(cosine_sim))
        else:
            for i in range(len(pos_idx)): # Somehow the old code is len(pos_idx) - 1, this means you always skip the last answer?!?
                start_idx = pos_idx[i]
                end_idx = pos_idx[i+1] if i < len(pos_idx)-1 else len(ranking_label)  
                subset = cosine_sim[start_idx: end_idx]
                output_softmax = F.softmax(subset, -1)
                label_conf = output_softmax[0]
                pred.append(torch.argmax(subset))
                label_confs.append(label_conf)
    pred = torch.tensor(pred)            # whatever is the argmax of the subset
    true_labels = torch.zeros_like(pred) # will always be the first one
    label_confs = torch.tensor(label_confs) 
    
    if return_label_conf:
        return pred, true_labels, label_confs
    return pred, true_labels