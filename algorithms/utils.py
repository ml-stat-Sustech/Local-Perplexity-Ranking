from .base_retrieve import BaseRetriever
from .topk import TopkRetriever


def get_retriever(retriever_type, task, ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device):
    if retriever_type == 'topk':
        print("topk")
        retriever = TopkRetriever(task, ice_dataloader, candidate_dataloader, noisy_model=noisy_model, noisy_tokenizer=noisy_tokenizer, device=device)
        
    else:
        print("Error Retriever")
    return retriever
    