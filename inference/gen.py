"""Direct Generation Inferencer"""

import json
import torch

from tqdm import tqdm
from accelerate import Accelerator
from typing import List, Union, Optional
from transformers import PretrainedConfig

from common.api import *
from algorithms import *
from .base_inferencer import BaseInferencer, GenInferencerOutputHandler
from common import PromptTemplate, get_logger, get_dataloader

logger = get_logger(__name__)

class GenInferencer(BaseInferencer):
    """Generation In-context Learning Inferencer Class
        In-context Learning Inferencer for Directly Generation.
        
    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file. 
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file. 
        api_name (:obj:`str`, optional): Name of API service. 
        call_api (:obj:`bool`): If ``True``, an API for LM models will be used, determined by :obj:`api_name`.   
        gen_field_replace_token (:obj:`str`, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the :obj:`model.generate()` method. 
    """

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.gen_field_replace_token = ''
        
        

    def inference(self,
                  task,
                  ice: Optional[List] = None,
                  candidate: Optional[List] = None,
                  labels: Optional[List] = None,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, normalizing_str: Optional[str] = None) -> List:

        self.generation_kwargs = {"max_new_tokens": 30}

        # 1. Preparation for output logs
        index = 0
        num = len(ice)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        
        # 3. Generate prompts for testing input 
        prompt_list = []
        
        for idx in range(len(ice)):
            prompt_list.append(str.replace(str.replace(str.replace(str.replace(ice_template[0], '</text>', candidate[idx]), '</E>', ice[idx]), "</answer>", ""), '"', ""))
    
        output_handler.save_orgin_prompts(prompt_list)
        # 4. Wrap prompts with Dataloader
        dataloader = get_dataloader(prompt_list, self.batch_size)

        # 5. Inference for prompts in each batch 
        logger.info("Starting inference process...")
        for entry in tqdm(dataloader, disable=not self.is_main_process):
            # 5-1. Inference with local model
            if not self.call_api:
                with torch.no_grad():
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    tokenized_data = self.tokenizer.batch_encode_plus(entry, padding=True, return_tensors='pt').to(self.device)
                    prompt_len = int(tokenized_data.attention_mask.shape[1])
                    
                    outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    outputs = outputs.tolist()
                    complete_output = self.tokenizer.batch_decode(outputs[:], skip_special_tokens=True)
                    generated = self.tokenizer.batch_decode([output[prompt_len:] for output in outputs],
                                                            skip_special_tokens=True)
            # 5-2. Inference with remote API
            else:
                complete_output, generated = api_get_tokens(self.api_name, entry)

            # 5-3. Save current output
            for prediction, output in zip(generated, complete_output):
                output_handler.save_prediction_and_output(prediction, output, index)
                index = index + 1

        # 6. Output 
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)
        return [sample['prediction'] for sample in output_handler.results_dict.values()]
