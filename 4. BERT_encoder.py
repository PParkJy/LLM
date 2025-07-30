from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentence = ["My favorite boy band is seventeen. The group is really good."]
inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
outputs = model(**inputs, output_attentions=True)

def plot_attention_matrix(matrix, title, tokens, ax):
    sns.heatmap(matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", cbar=False, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)

# 함수 1. 한 layer 내의 여러 head 비교
def visualize_heads_in_layer(layer_idx=0):
    layer = attentions[layer_idx][0].detach().cpu().numpy()  # (heads, seq_len, seq_len)
    num_heads = layer.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(3*num_heads, 3))
    fig.suptitle(f"Layer {layer_idx} - All Heads", fontsize=14)
    for h in range(num_heads):
        plot_attention_matrix(layer[h], f"Head {h}", tokens, axes[h])
    plt.tight_layout()
    plt.show()

# 함수 2. 한 head의 여러 layer 비교
def visualize_same_head_across_layers(head_idx=0):
    num_layers = len(attentions)
    fig, axes = plt.subplots(1, num_layers, figsize=(3*num_layers, 3))
    fig.suptitle(f"Head {head_idx} Across Layers", fontsize=14)
    for l in range(num_layers):
        attn = attentions[l][0, head_idx].detach().cpu().numpy()
        plot_attention_matrix(attn, f"Layer {l}", tokens, axes[l])
    plt.tight_layout()
    plt.show()

# 함수 3. 입력 token이 어떤 token에 집중했는지 판단
def trace_attention_for_token(target_token: str):
    token_list = tokenizer.convert_ids_to_tokens(input_ids[0])
    # 중복인 경우 첫 번째 토큰 선택
    try:
        target_idx = token_list.index(target_token)
    except ValueError:
        print(f"Token '{target_token}' not found in: {token_list}")
        return

    print(f"\n Token: '{target_token}' at position {target_idx}")
    print(f"{'Layer':<6} {'Head':<6} Top-5 Attended Tokens")
    print("-" * 40)
    
    for layer_idx, layer_attn in enumerate(attentions):  # layer_attn: [1, num_heads, seq_len, seq_len]
        attn = layer_attn[0]  # shape: [num_heads, seq_len, seq_len]
        for head_idx in range(attn.shape[0]):
            vector = attn[head_idx, target_idx].detach().cpu().numpy()
            scores = [(token_list[i], round(float(vector[i]), 3)) for i in range(len(token_list))]
            top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
            formatted = ", ".join([f"{tok}:{val}" for tok, val in top5])
            print(f"{layer_idx:<6} {head_idx:<6} {formatted}")

## 1. Choose target layer and head ===================================================================================                              
target_layer = 11  # Layer 0 - 11
head_idx = 0 # 0 - 11
sa = model.encoder.layer[target_layer].attention.self

## 2. Hidden states from embedding ===================================================================================
with torch.no_grad():
    embedding_output = model.embeddings(input_ids)

## 3. Linear projections ===================================================================================
Q = sa.query(embedding_output)
K = sa.key(embedding_output)
V = sa.value(embedding_output)

## 4. Shape and Reshape information ===================================================================================
batch_size, seq_len, hidden_dim = Q.shape
num_heads = sa.num_attention_heads
head_dim = hidden_dim // num_heads

def reshape_to_heads(x):
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # Reshape to [batch, num_heads, seq_len, head_dim]

Q = reshape_to_heads(Q)
K = reshape_to_heads(K)
V = reshape_to_heads(V)

## 5. Attention score: Q x K^T / sqrt(d) ===================================================================================
attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)

## 6. Apply mask (optional) ===================================================================================
extended_mask = attention_mask[:, None, None, :]  # shape: [batch, 1, 1, seq_len]
extended_mask = (1.0 - extended_mask) * -10000.0
attention_scores += extended_mask

## 7. Softmax → attention weights ===================================================================================
attention_probs = F.softmax(attention_scores, dim=-1)

## 8. Weighted sum ===================================================================================
context = torch.matmul(attention_probs, V)  # shape: [batch, num_heads, seq_len, head_dim]

## 9. Concatenate heads ===================================================================================
context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

## 10. Print attention at one head ===================================================================================
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"\n[Layer {target_layer}, Head {head_idx}] Attention Matrix :")
att_matrix = attention_probs[0, head_idx].detach().cpu().numpy()

for i, row in enumerate(att_matrix):
    attn_to = [(tokens[j], round(float(row[j]), 3)) for j in range(len(tokens))]
    topk = sorted(attn_to, key=lambda x: x[1], reverse=True)[:10]
    print(f"Token '{tokens[i]}' attends most to: {topk}")

## 11. Show attention ===================================================================================
attentions = outputs.attentions 
visualize_heads_in_layer(target_layer) # 각 head가 어떤 패턴으로 context를 수집했는지   
visualize_same_head_across_layers(head_idx) 

trace_attention_for_token("seventeen")
trace_attention_for_token("[CLS]")      

## 12. Show context vector
seventeen_idx = tokens.index("seventeen")
print("Context vector for 'seventeen' (last layer):")
print(context[0, seventeen_idx])  