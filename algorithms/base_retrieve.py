"""Basic Retriever"""
import copy
import math
import faiss
import torch
import itertools
import numpy as np

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer


from typing import List
from transformers import AutoTokenizer



from common import get_prompt_label, DataCollatorWithPaddingAndCuda, DatasetEncoder, extract_data
class BaseRetriever:
    index_ds = None
    test_ds = None

    def __init__(self, task, ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device):
        self.task = task
        self.index_ds = extract_data(ice_dataloader, self.task)
        self.test_ds =  extract_data(candidate_dataloader, self.task)
        
        self.template, self.template_dict, self.label = get_prompt_label(self.task)
        self.device = device
        
        self.tokenizer_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = SentenceTransformer(self.tokenizer_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.noise_classifier_model = noisy_model
        self.noise_classifier_tokenizer = noisy_tokenizer
        self.noise_classifier_model.eval()
        
        self.batch_size = 8
        self.test_text = self.test_ds['text'].tolist()
        self.ice_text = self.index_ds['text'].tolist()
        self.test_encode_dataset = DatasetEncoder(self.test_text, tokenizer=self.tokenizer)
        self.ice_encode_dataset = DatasetEncoder(self.ice_text, tokenizer=self.tokenizer)
        self.co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        self.test_dataloader = DataLoader(self.test_encode_dataset, batch_size=self.batch_size, collate_fn=self.co)
        self.index_dataloader = DataLoader(self.ice_encode_dataset, batch_size=self.batch_size, collate_fn=self.co)
        self.ice_forward = self.forward(self.index_dataloader, process_bar=True, information="Embedding ice set...")
        self.text_forward = self.forward(self.test_dataloader, process_bar=True, information="Embedding test set...")

        self.base_index = self.create_base_index(self.index_dataloader)
        self.scale_factor = 0.1


    def retrieve(self, ice_num, noise_retriever_type, knn_num, tau) -> List[List]:
        """
            Retrieve for each data in generation_ds.
            
        Returns:
            `List[List]`: the index list of in-context example for each data in `test_ds`.
        """
        raise NotImplementedError("Method hasn't been implemented yet")


    def noise_retrieve(self, noise_retriever_type, embed, idx_list, ice_num, knn_num, tau):
        if noise_retriever_type == True:
            ppl = []
            new_can_list = []
            for ice_idx in idx_list:
                test = next(itertools.islice(self.ice_forward, ice_idx, None))
                test_embed = np.expand_dims(test['embed'], axis=0)
                knn_list = self.base_index.search(test_embed, 16)[1][0].tolist()
                res_list = self.check_list(new_can_list, knn_list, knn_num, self.device)
                score_list = []

                for idx, res_idx in enumerate(res_list):
                    text = str.replace(str.replace(str(self.template), '</answer>', ""), '</text>', self.index_ds['text'][res_idx])
                    ppl_score = self.perplexity(text+str(self.index_ds['label'][res_idx]), self.noise_classifier_model, self.noise_classifier_tokenizer, self.device)
                    score_list.append(ppl_score)

                if score_list[0] < np.percentile(score_list, tau):
                    new_can_list.append(res_list[0])
                else:
                    new_can_list.append(res_list[score_list.index(min(score_list))])


            # ordered by relevance score
            near_reps, rel_scores, kernel_matrix = self.get_kernel(embed, new_can_list)
            samples_ids = self.fast_map_dpp(kernel_matrix, ice_num)
            samples_scores = np.array([rel_scores[i] for i in samples_ids])
            samples_ids = np.array(samples_ids)[(-samples_scores).argsort()].tolist()
            rtr_list = [int(new_can_list[i]) for i in samples_ids]

            
            
        elif  noise_retriever_type == False:
            rtr_list = idx_list
            
        else: 
            rtr_list = False

        return rtr_list

    
    def create_base_index(self, dataloader):
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))
        res_list = self.forward(dataloader)
        id_list = np.array([res['metadata']['id'] for res in res_list])
        self.embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(self.embed_list, id_list)
        return index
    

    def forward(self, dataloader, process_bar=False, information=''):
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
                res = self.model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list
    
    
    def check_list(self, candidate_ice_list, referance_ice_list, knn_num,device):
        candidate_ice_tensor = torch.tensor(candidate_ice_list).to(device)
        referance_ice_tensor = torch.tensor(referance_ice_list).to(device)
        tensor_diff = torch.isin(referance_ice_tensor,candidate_ice_tensor, invert=True)
        checklist = np.array(referance_ice_tensor[tensor_diff].cpu()).tolist()
        return checklist[0:knn_num]
    
    
    def perplexity(self, text, model, tokenizer, device, max_length = 500, stride = 512):
        tokenizer.pad_token = tokenizer.eos_token
        encodings = tokenizer(text, padding=True, return_tensors='pt', truncation=True, max_length=500)
        seq_len = encodings.input_ids.size(1)   

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return np.array(ppl.cpu())
    
    ####Demonstration Reorder########
    def get_kernel(self, embed, candidates):
        near_reps = np.stack([self.base_index.index.reconstruct(i) for i in candidates], axis=0)
        # normalize first
        embed = embed / np.linalg.norm(embed)
        near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

        # to make kernel-matrix non-negative
        rel_scores = np.matmul(embed, near_reps.T)[0]
        rel_scores = (rel_scores + 1) / 2

        # to prevent overflow error
        rel_scores -= rel_scores.max()

        # to balance relevance and diversity
        rel_scores = np.exp(rel_scores / (2 * self.scale_factor))

        # to make kernel-matrix non-negative
        sim_matrix = np.matmul(near_reps, near_reps.T)
        sim_matrix = (sim_matrix + 1) / 2

        kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
        return near_reps, rel_scores, kernel_matrix
    

    ####Demonstration Reorder########
    def fast_map_dpp(self, kernel_matrix, max_length):
        item_size = kernel_matrix.shape[0]
        cis = np.zeros((max_length, item_size))
        di2s = np.copy(np.diag(kernel_matrix))
        selected_items = list()
        selected_item = np.argmax(di2s)
        selected_items.append(int(selected_item))
        while len(selected_items) < max_length:
            k = len(selected_items) - 1
            ci_optimal = cis[:k, selected_item]
            di_optimal = math.sqrt(di2s[selected_item])
            elements = kernel_matrix[selected_item, :]
            eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
            cis[k, :] = eis
            di2s -= np.square(eis)
            selected_item = np.argmax(di2s)
            selected_items.append(int(selected_item))
        return selected_items
    






