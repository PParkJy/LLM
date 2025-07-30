from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

# CLS vector가 전체 문장의 context를 반영하고 있는지 확인

# 1: Load model & tokenizer ====================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

# 2: Define sentences ====================
sentences = [
    "My favorite boy band is seventeen.",
    "I enjoy listening to music from K-pop groups.",
    "The weather today is sunny and bright.",
    "Seventeen is a talented boy band.",
    "Dogs are very loyal and friendly animals.",
    "I am a K-pop music fan."
]

# 3: Extract CLS vectors ====================
cls_vectors = []
tokenized_sentences = []

with torch.no_grad():
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        outputs = model(**inputs)
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze()  # [768]
        cls_vectors.append(cls_vec)
        tokenized_sentences.append(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

print("\n[CLS] vectors (first 10 dimensions):")
for i, vec in enumerate(cls_vectors):
    print(f"[{i}] {sentences[i]}")
    print(vec[:10])  

# 4: Compute cosine similarity ====================
print("\nCosine similarity between [CLS] vectors:")
for i in range(len(sentences)):
    for j in range(len(sentences)):
        sim = F.cosine_similarity(cls_vectors[i].unsqueeze(0), cls_vectors[j].unsqueeze(0)).item()
        print(f"  ({i}) vs ({j}) : {sim:.4f}")

# 5: Query-based ranking ====================
query_idx = 0  # Use sentence 0 as the query
query_vec = cls_vectors[query_idx]

similarities = [
    (i, F.cosine_similarity(query_vec.unsqueeze(0), vec.unsqueeze(0)).item())
    for i, vec in enumerate(cls_vectors)
]
similarities.sort(key=lambda x: x[1], reverse=True)

print(f"\nQuery: {sentences[query_idx]}")
print("\nTop similar sentences:")
for idx, score in similarities:
    print(f"  Score: {score:.4f} | Sentence: {sentences[idx]}")

# T-SNE Visualization ====================
cls_np = torch.stack(cls_vectors).cpu().numpy()
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
cls_2d = tsne.fit_transform(cls_np)

plt.figure(figsize=(10, 6))
for i, sent in enumerate(sentences):
    x, y = cls_2d[i]
    color = "red" if i == query_idx else "blue"
    plt.scatter(x, y, color=color)
    plt.text(x + 0.3, y + 0.3, f"{i}", fontsize=12)
plt.title("t-SNE of [CLS] embeddings (Query in red)")
plt.grid(True)
plt.show()