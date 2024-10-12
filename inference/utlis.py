from .base_inferencer import BaseInferencer
from .gen import GenInferencer

def get_inferencer(inferencer_type, model_name, tokenizer_name, batch_size=16):
    inferencer = GenInferencer(model_name=model_name, tokenizer_name=tokenizer_name, batch_size=batch_size)
    return inferencer