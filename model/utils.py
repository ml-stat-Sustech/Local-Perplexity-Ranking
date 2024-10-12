from transformers import AutoTokenizer, AutoModelForCausalLM

pretrained_model_dic = {
        "llama":"Llama-2-7b-chat-hf",
        }


def get_model(pretrained_model_name):
    if pretrained_model_name in pretrained_model_dic:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dic[pretrained_model_name],torch_dtype='auto')
        print(pretrained_model_dic[pretrained_model_name])
    else:
        print("Error: Model Type")
        
    return model


def get_tokenizer(pretrained_model_name):

    if pretrained_model_name in pretrained_model_dic:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dic[pretrained_model_name], padding_side='left', padding=True, return_tensors='pt', truncation=True, max_length=2048)

    else:
        print("Error: Tokenizer Type")
        
    return tokenizer