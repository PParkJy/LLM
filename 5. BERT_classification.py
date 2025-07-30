from transformers import BertConfig, BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
model = BertModel.from_pretrained("bert-base-uncased")

sequence= ["My favorite boy band is seventeen. The group is really good."]

inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)

print("last_hidden_state (shape):", outputs.last_hidden_state.shape)
for idx, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])):
    vec = outputs.last_hidden_state[0, idx]  # [768]
    print(f"[{idx}] Token: '{token}' → Vector: {vec.tolist()[:5]}...") 

print("\n pooler_output (shape):", outputs.pooler_output.shape)
print("):", outputs.pooler_output[0].tolist()[:10], "...")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Input sentence
inputs = tokenizer("My favorite boy band is seventeen. The group is really good.", return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Extract vectors
cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()      # [CLS]
pooler_vec = outputs.pooler_output.squeeze().cpu().numpy()                # Pooler output

# Plot
plt.figure(figsize=(14, 4))
plt.plot(cls_vec, label="[CLS] from last_hidden_state", alpha=0.7)
plt.plot(pooler_vec, label="pooler_output (after Linear + tanh)", alpha=0.7)
plt.title("Comparison: [CLS] vs pooler_output")
plt.xlabel("Hidden dimension (0 ~ 767)")
plt.ylabel("Activation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()