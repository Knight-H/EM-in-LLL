from pytorch_transformers import BertModel
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import pickle
import os
import random
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


from settings import model_classes
from utils import pad_to_max_len


class MemoryFewRel:
    def __init__(self, args):
        self.n_neighbors = args.n_neighbors
        with torch.no_grad():
            logger.info("Initializing memory {} model".format(args.model_name))
            self.model = BertModel.from_pretrained(args.model_name).cuda()
            self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self.max_len = self.model.config.max_position_embeddings
        self.keys, self.texts, self.labels, self.candidates = [], [], [], []
        self.tree = NearestNeighbors(n_jobs=args.n_workers)
        self.built_tree = False

    # Update keys, input_ids, and labels --> Change to text, label, candidate
    def add(self, texts, labels, candidates, input_ids, masks, write_prob=1.):
        # If not in range, ignore write_prob
        if random.random() > write_prob:
            return
        if self.built_tree:
            logging.warning("Tree already build! Ignore add.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        self.keys.extend(outputs[0][:, 0, :].detach().cpu().tolist())
        self.texts.extend(texts)
        self.labels.extend(labels)
        self.candidates.extend(candidates)
        #print(f"Successful add. Total Keys: {len(self.keys)}. Total Texts: {len(self.texts)}. Total labels: {len(self.labels)}. Total cand {len(self.candidates)}")
        del outputs

    #
    def sample(self, n_samples):
        if self.built_tree:
            logging.warning("Tree already build! Ignore sample.")
            return
        inds = np.random.randint(len(self.labels), size=n_samples)
        texts = [self.texts[ind] for ind in inds]
        labels = [self.labels[ind] for ind in inds]
        candidates = [self.candidates[ind] for ind in inds]
        # input_ids, masks = pad_to_max_len(input_ids)
        # labels = torch.tensor(labels, dtype=torch.long)
        # return input_ids.cuda(), masks.cuda(), labels.cuda()
        return texts, labels, candidates

    # Removed NP Array in input_ids and labels, does it make any difference?
    def build_tree(self):
        if self.built_tree:
            logging.warning("Tree already build! Ignore build.")
            return
        self.built_tree = True
        self.keys = np.array(self.keys)
        self.tree.fit(self.keys)
        # self.input_ids = np.array(self.input_ids, dtype=object)
        # self.labels = np.array(self.labels)

    # Knight Edit: make self.input_ids and self.labels do np.array on the fly
    def query(self, input_ids, masks):
        #if not self.built_tree:
        #    logging.warning("Tree not built! Ignore query.")
        #    return
        outputs = self.model(input_ids=input_ids, attention_mask=masks)
        queries = outputs[0][:, 0, :].cpu().numpy()
        inds = self.tree.kneighbors(queries, n_neighbors=self.n_neighbors, return_distance=False)
#         print("inds", inds)
#         print("keys", np.array(self.keys).shape)
#         print("np.array(self.texts, dtype=object)", np.array(self.texts, dtype=object).shape)

        texts_batch = [text_batch.tolist() for text_batch in np.array(self.texts, dtype=object)[inds]]
        labels_batch = [label_batch.tolist() for label_batch in np.array(self.labels, dtype=object)[inds]]
        candidates_batch = [candidate_batch.tolist() for candidate_batch in np.array(self.candidates, dtype=object)[inds]]
        
        # labels_batch = [torch.tensor(label, dtype=torch.long) for label in np.array(self.labels)[inds]]
        # candidates_batch = [torch.tensor(candidate, dtype=torch.long) for candidate in np.array(self.candidates)[inds]]
        return texts_batch, labels_batch, candidates_batch
