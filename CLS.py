from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

# 1: Load model & tokenizer ====================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True, output_hidden_states=True)
model.eval()

# 2: Define sentence ====================
sentence = "My favorite boy band is seventeen. The group is really good."
inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# 3: Embedding visualization ====================
with torch.no_grad():
    word_embeds = model.embeddings.word_embeddings(input_ids)
    pos_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
    pos_embeds = model.embeddings.position_embeddings(pos_ids)
    token_type_ids = torch.zeros_like(input_ids)
    type_embeds = model.embeddings.token_type_embeddings(token_type_ids)
    sum_embed = word_embeds + pos_embeds + type_embeds

    print("\n[CLS] Token Embedding Components (first 10 dims):")
    print("Word:", word_embeds[0, 0, :10])
    print("Positional:", pos_embeds[0, 0, :10])
    print("TokenType:", type_embeds[0, 0, :10])
    print("Sum:", sum_embed[0, 0, :10])

# 4: Forward pass ====================
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    attentions = outputs.attentions
    hidden_states = outputs.hidden_states

# 5: Plot attention heads for CLS ====================
def plot_cls_attention(layer_idx):
    attn = attentions[layer_idx][0]  # [heads, seq_len, seq_len]
    cls_attn = attn[:, 0, :]  # [heads, seq_len]
    fig, axes = plt.subplots(1, cls_attn.shape[0], figsize=(3*cls_attn.shape[0], 3))
    for h in range(cls_attn.shape[0]):
        sns.heatmap(cls_attn[h].cpu().numpy().reshape(1, -1), xticklabels=tokens,
                    yticklabels=[f"Head {h}"], ax=axes[h], cbar=False, cmap="viridis")
    plt.suptitle(f"[CLS] Attention in Layer {layer_idx}")
    plt.tight_layout()
    plt.show()

plot_cls_attention(layer_idx=11)

# 6: Plot all layer attention maps ====================
def plot_all_layer_attention():
    for layer_idx in range(len(attentions)):
        attn = attentions[layer_idx][0]  # [heads, seq_len, seq_len]
        avg_attn = attn.mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
        plt.figure(figsize=(8, 6))
        sns.heatmap(avg_attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
        plt.title(f"Average Attention Map (Layer {layer_idx})")
        plt.tight_layout()
        plt.show()

plot_all_layer_attention()

# 7: Track CLS vector across layers ====================
cls_trajectory = [hs[0, 0, :].cpu().numpy() for hs in hidden_states]
sims = [F.cosine_similarity(
    torch.tensor(cls_trajectory[0]).unsqueeze(0),
    torch.tensor(vec).unsqueeze(0), dim=-1).item() for vec in cls_trajectory]

plt.figure(figsize=(8, 4))
plt.plot(range(len(sims)), sims, marker="o")
plt.title("Cosine Similarity of [CLS] Vector Across Layers")
plt.xlabel("Layer")
plt.ylabel("Similarity to Embedding")
plt.grid(True)
plt.show()

# 8: Show final CLS vector ====================
print("\nFinal [CLS] vector (last layer, first 10 dims):")
print(last_hidden_state[0, 0, :10])

# 9: Trace attention for specific token ====================
def trace_token_attention(token_str):
    try:
        token_idx = tokens.index(token_str)
    except ValueError:
        print(f"Token '{token_str}' not found.")
        return

    print(f"\nTracing attention to '{token_str}' (index {token_idx}) across layers")
    for layer_idx in range(len(attentions)):
        attn = attentions[layer_idx][0]  # [heads, seq_len, seq_len]
        for head in range(attn.shape[0]):
            weights = attn[head, :, token_idx]  # who is attending TO this token?
            topk = torch.topk(weights, k=3).indices.tolist()
            print(f"Layer {layer_idx}, Head {head} → Top tokens attending TO '{token_str}': {[tokens[i] for i in topk]}")

trace_token_attention("seventeen")
trace_token_attention("group")

# 10: V vector weighted sum for CLS ====================
# Extract layer 11 self-attention
sa_module = model.encoder.layer[11].attention.self

with torch.no_grad():
    embedding_output = model.embeddings(input_ids)
    Q = sa_module.query(embedding_output)
    K = sa_module.key(embedding_output)
    V = sa_module.value(embedding_output)

    num_heads = sa_module.num_attention_heads
    head_dim = sa_module.attention_head_size

    def reshape(x):
        return x.view(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)

    Q = reshape(Q)
    K = reshape(K)
    V = reshape(V)

    scores = torch.matmul(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    weighted_V = torch.matmul(probs, V)  # [1, heads, seq_len, head_dim]
    context_vec = weighted_V.permute(0, 2, 1, 3).contiguous().view(1, -1, num_heads * head_dim)

    print("\n[CLS] context vector (layer 11, after attention):")
    print(context_vec[0, 0, :10])
