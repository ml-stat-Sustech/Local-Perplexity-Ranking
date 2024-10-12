import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import torch
import argparse
import evaluate

from inference import get_inferencer
from algorithms import get_retriever
from dataset.utils import get_dataloader
from model import get_model, get_tokenizer
from common import get_prompt_label, extract_data, setup_seed, get_input, delect_unavailable_word


def main(args):

    setup_seed(100)
    device = torch.device("cuda:0")
    model, tokenizer = get_model(args.pretrained_model_name).to(device), get_tokenizer(args.pretrained_model_name)

    em = []
    for seed in [100]:
        setup_seed(seed)

        model.eval()
        #####get data#####
        raw_ice_dataset = get_dataloader(args.task, 'train', args.ice_noisy_type, args.noisy_ratio, seed)
        raw_test_dataset = get_dataloader(args.task, 'test', 'real', args.noisy_ratio, seed)
        raw_ice_dataloader = torch.utils.data.DataLoader(raw_ice_dataset, batch_size=1, shuffle=False)
        raw_test_dataloader = torch.utils.data.DataLoader(raw_test_dataset, batch_size=1, shuffle=False)
        ice_dataset = extract_data(raw_ice_dataloader, args.task)
        test_dataset = extract_data(raw_test_dataloader, args.task)

        #####get in-context demonstration#####
        retriever = get_retriever(args.test_retrieving, args.task, raw_ice_dataloader, raw_test_dataloader, model, tokenizer, device)
        ice_idx_list = retriever.retrieve(args.ice_num, args.noise_retrieving, args.knn_num, args.tau)
        template, template_dict, label = get_prompt_label(args.task)
        ice = get_input(args.task, ice_idx_list, template, template_dict, ice_dataset)
        
        #####Inference#####
        inferencer = get_inferencer('gen', model_name=model, tokenizer_name = tokenizer, batch_size=args.batch_size)
        test_predictions = inferencer.inference(task=args.task,ice=ice,  candidate=test_dataset['text'], labels=list(range(len(label))), ice_template=template_dict)
        preds = delect_unavailable_word(test_predictions)
        
        #####Evaluate#####
        em_evaluate = evaluate.load('exact_match')
        em.append(em_evaluate.compute(predictions=preds, references=test_dataset['label']))
    print(em)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #task and prompt
    parser.add_argument('--task', type=str, choices=['sciq'], default='sciq', help='task.')
    
    #retriever
    parser.add_argument('--test_retrieving', type=str, choices=['random', 'topk', 'dpp', 'zero'], default='topk', help='Choose demonstration selection method.')
    parser.add_argument('--noise_retrieving', type=bool, choices=[True, False], default=True, help='Choose noise retriever.')
    parser.add_argument('--ice_num',  type=int, default=4)

    parser.add_argument('--tau',  type=int, choices=[25, 50, 75], default=50)
    parser.add_argument('--knn_num',  type=int, choices=[2, 4, 6, 8], default=4)

    #noise label
    parser.add_argument('--noisy_ratio', type=float, default=0.6, help='noisy ratio.')
    parser.add_argument('--ice_noisy_type', type=str, choices=['irrelevant', "relevant", "real"], default="relevant", help='noisy type.')

    #model
    parser.add_argument('--pretrained_model_name', '-m', choices=['llama'], type=str, default='llama', help='Choose pretrained model.')
    
    #others
    parser.add_argument('--batch_size', type=int, default=4, help='Test batch size.')
    args = parser.parse_args()
    main(args)

