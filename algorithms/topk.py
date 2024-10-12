"""Topk Retriever"""
import tqdm
import numpy as np

from common import get_logger
from .base_retrieve import BaseRetriever

logger = get_logger(__name__)



class TopkRetriever(BaseRetriever):
    model = None
    def __init__(self, task, ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device):
        super().__init__(task, ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device)
        
    def knn_search(self, ice_num, noise_retriever_type, knn_num, knn_q):
        rtr_idx_list = [[] for _ in range(len(self.text_forward))]
        logger.info("Retrieving data for test set...")

        for entry in tqdm.tqdm(self.text_forward):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.base_index.search(embed, ice_num)[1][0].tolist()
            rtr_idx_list[idx] = self.noise_retrieve(noise_retriever_type, embed, near_ids, ice_num, knn_num, knn_q)
        return rtr_idx_list


    def retrieve(self, ice_num, noise_retriever_type, knn_num, knn_q):
        return self.knn_search(ice_num, noise_retriever_type, knn_num, knn_q)
