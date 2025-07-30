from transformers import BertConfig, BertTokenizer, BertModel
import torch
import inspect

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
model = BertModel.from_pretrained("bert-base-uncased")

sequence= ["My favorite boy band is seventeen. The group is really good."]

inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)

## 1. What is input of BERT (docs) =======================================================================                              
# print(model.forward.__doc__)
# print(inspect.signature(model.forward))

## 2. What is input of BERT (real input) =======================================================================                              
# sig = inspect.signature(model.forward)
# arg_names = list(sig.parameters.keys())

# print("=== Model forward inputs ===")
# for name in arg_names:
#     if name in inputs:
#         print(f"{name}:\n{inputs[name]}\n") #input_ids, attention_mask, token_type_ids
#     #else:
#     #    print(f"{name}: Use default of BERT\n")
