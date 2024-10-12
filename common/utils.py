import torch
import random
import numpy as np
import pandas as pd


from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from typing import List

class DatasetEncoder(torch.utils.data.Dataset):
    def __init__(self, datalist: List, model_name=None, tokenizer=None) -> None:
        self.datalist = datalist
        if model_name is None and tokenizer is None:
            raise ValueError("model_name and tokenizer could not both be None")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
        self.encode_dataset = []
        self.init_dataset()
        self.datalist_length = len(self.encode_dataset)

    def init_dataset(self):
        for idx, data in enumerate(self.datalist):
            tokenized_data = self.tokenizer.encode_plus(data, truncation=True, return_tensors='pt', verbose=False)
            self.encode_dataset.append({
                'input_ids': tokenized_data.input_ids[0],
                'attention_mask': tokenized_data.attention_mask[0],
                "metadata": {"id": idx, "len": len(tokenized_data.input_ids[0]),
                             "text": data}
            })

    def __len__(self):
        return self.datalist_length

    def __getitem__(self, idx):
        return self.encode_dataset[idx]

        

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    



def get_inputes(task, new_ice_idx, new_ice_target, template, template_dict, training_dataset):
    generated_ice_list = []
    for idx_list in new_ice_idx:
        raw_ice = ''
        for idx in idx_list:
            raw_ice = raw_ice + str.replace(str.replace(template, '</text>', str(training_dataset['text'][idx])), '</answer>', str(new_ice_target[idx])) + '\n'
        generated_ice_list.append(raw_ice)

    return generated_ice_list


def collote_fn(batch_samples, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)):
    batch_sentence = []
    batch_label = []
    for text, target, index in batch_samples:
        batch_sentence.append(text)
        batch_label.append(target)
        
    X = tokenizer(
        batch_sentence, 
        padding=True,
        max_length = 512, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y



def get_prompt_label(task):
    if task == 'sciq':
        template = '</E> \n </text> \n Output: </answer>'
        labels = []
        template_dict = {0:'</E> \n </text> \n Output: </answer>'}
    

    else:
        print('ERROR PROMPT')
    
    return template, template_dict, labels


def extract_data(dataloader, task):
    texts = []
    labels = []
    for text, target, index in tqdm (dataloader):
        texts.append(text[0])
        labels.append(target[0])

    data = pd.DataFrame({"text": texts, "label": labels})
    return data


def generate_label_prompt(idx, test_ds, ice, label, template):
    raw_text = str.replace(template[label], '</text>', test_ds)
    prompt = str.replace(raw_text, '</E>', ice)
    return prompt


def get_input(task, ice_idx_list, template, template_dict, training_dataset):
    generated_ice_list = []
    for idx_list in ice_idx_list:
        raw_ice = ''
        for idx in idx_list:
            raw_ice = raw_ice + str.replace(str.replace(template, '</text>', str(training_dataset['text'][idx])), '</answer>', str(training_dataset['label'][idx])) + '\n'
        generated_ice_list.append(raw_ice)
            
    return generated_ice_list



def delect_unavailable_word(text):
    preds = []
    for pred in tqdm(text):
        preds.append(str.replace(str.replace(pred.split('Support', 1)[0], '</E>', ''),'\n','').strip())
    return preds


def get_dataloader(datalist: List[List], batch_size: int) -> DataLoader:
    dataloader = DataLoader(datalist, batch_size=batch_size)
    return dataloader