from transformers import BertConfig, BertTokenizer, BertModel
import torch
import inspect
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
model = BertModel.from_pretrained("bert-base-uncased")

sequence= ["My favorite boy band is seventeen. The group is really good."]

inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)

'''
Embedding
	(word_embeddings): Embedding(30522, 768, padding_idx=0)
	(position_embeddings): Embedding(512, 768)
	(token_type_embeddings): Embedding(2, 768)
	(LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
	(dropout): Dropout(p=0.1, inplace=False)

Pooling
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
'''

input_ids = inputs["input_ids"]               # [1, seq_len]
token_type_ids = inputs["token_type_ids"]     # [1, seq_len]
seq_len = input_ids.shape[1]
position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(input_ids)

## 1. Embeddings ===================================================================================                              
### 1. Individual embeddings =======================================================================                              
word_embeds = model.embeddings.word_embeddings(input_ids)
pos_embeds = model.embeddings.position_embeddings(position_ids)
token_type_embeds = model.embeddings.token_type_embeddings(token_type_ids)

### 2. Sum before LayerNorm =======================================================================                              
sum_embed = word_embeds + pos_embeds + token_type_embeds

### 3. After LayerNorm =======================================================================                              
layernorm = model.embeddings.LayerNorm
normed_embed = layernorm(sum_embed)

### 4. After (only valid in training) =======================================================================                              
dropout = model.embeddings.dropout
final_embed = dropout(normed_embed)

### 5. Result =======================================================================     
dim = 10# embedding dimension                         
print(f"\n=== Embedding analysis (dim={dim}) ===")
for token_index in range(seq_len):
    token_id = input_ids[0, token_index].item()
    token_str = tokenizer.convert_ids_to_tokens(token_id)
    print(f"\n--- Token [{token_index}]: '{token_str}' ---")
    print("Word Embedding:        ", word_embeds[0, token_index, :dim])
    print("Position Embedding:    ", pos_embeds[0, token_index, :dim])
    print("Token Type Embedding:  ", token_type_embeds[0, token_index, :dim])
    print("Sum Before LayerNorm:  ", sum_embed[0, token_index, :dim])
    print("After LayerNorm:       ", normed_embed[0, token_index, :dim])
    print("After Dropout:         ", final_embed[0, token_index, :dim])
    
### 6. Visualization =======================================================================
#### 1. Final_vector extraction
final_embed_np = final_embed[0].detach().cpu().numpy()  # shape: [seq_len, hidden_dim]
token_ids = input_ids[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(token_ids)

#### 2. T-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embed_2d = tsne.fit_transform(final_embed_np)  # shape: [seq_len, 2]

#### 3. Show graph
plt.figure(figsize=(10, 6))
for i, token in enumerate(tokens):
    x, y = embed_2d[i]
    plt.scatter(x, y, color='green')
    plt.text(x + 0.3, y + 0.3, token, fontsize=12)

plt.title("BERT Final Embedding (t-SNE 2D)")
plt.grid(True)
plt.show()                             

## 2. Pooler ===================================================================================
with torch.no_grad():
    pooled_output = model.pooler(model(**inputs).last_hidden_state) #last_hidden_state = [CLS]
    #print(pooled_output.shape)
    
    print(pooled_output[0, :dim])  