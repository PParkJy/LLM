from transformers import BertConfig, BertTokenizer, BertModel
import torch

## 0. BERT =======================================================================
# https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/bert/modeling_bert.py

## 1. Configuration =======================================================================
'''
class BertConfig(PretrainedConfig):
		def __init__(
	        self,
	        vocab_size=30522, # 고유 토큰 개수
	        hidden_size=768, # Embedding vector 차원
	        num_hidden_layers=12, # Encoder hidden layer 개수 (=Encoder stack 개수)
	        num_attention_heads=12, # Attention head 개수
	        intermediate_size=3072, # FFN network 차원
	        hidden_act="gelu", # Activation function
	        hidden_dropout_prob=0.1, # FC layer들의 dropout 비율
	        attention_probs_dropout_prob=0.1, # Attention 확률의 dropout 비율
	        max_position_embeddings=512, # 모델에 입력가능한 최대 token 개수
	        type_vocab_size=2, #
	        initializer_range=0.02,#
	        layer_norm_eps=1e-12, # Layer normalization 
	        pad_token_id=0,
	        position_embedding_type="absolute", # Position embedding -> absolute, relative_key, relative_key_query
	        use_cache=True,
	        classifier_dropout=None, # Classification head의 dropout 비율
	        **kwargs
	    ):
'''

# config = BertConfig.from_pretrained("bert-base-uncased")
# print(config)

## If want to fix the config...
# config = BertConfig.from_pretrained("bert-base-uncased")
# config.output_hidden_states = True 
# config.hidden_size = 518 # -> Pre-trained 결과와 동일x: 오류
# model = BertModel.from_pretrained("bert-base-uncased", config=config)


## 2. Basic model (Load model)=======================================================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
model = BertModel.from_pretrained("bert-base-uncased")

#inputs = tokenizer("My favorite boy band is seventeen", return_tensors="pt") 
#outputs = model(**inputs)
#last_hidden_states = outputs.last_hidden_state


## 3. Tokenizer =======================================================================
### Input embedding
# sequence= ["My favorite boy band is seventeen. The group is really good."]
# inputs = tokenizer(sequence, return_tensors="pt")
  
# outputs = model(**inputs)
# input_ids = inputs['input_ids'] 
# print(input_ids) # tensor([[ 101, 2026, 5440, 2879, 2316, 2003, 9171, 1012, 1996, 2177, 2003, 2428, 2204, 1012,  102]])

# decoded = tokenizer.decode(input_ids[0])
# print(decoded) # [CLS] my favorite boy band is seventeen. the group is really good. [SEP]

# ### For NSP
# seq_a = "My favorite boy band is seventeen."
# seq_b = "Who is the my favorite band?"

# encoded_dict = tokenizer(seq_a, seq_b)
# encoded_dict["token_type_ids"] 
# print(encoded_dict["token_type_ids"]) # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# decoded = tokenizer.decode(encoded_dict["input_ids"])
# print(decoded) # [CLS] my favorite boy band is seventeen. [SEP] who is the my favorite band? [SEP]


## 3. BERT model =======================================================================
print(model) # Embedding -> Encoder -> Pooler (Linear layer)
'''
(embeddings): BertEmbeddings(
	(word_embeddings): Embedding(30522, 768, padding_idx=0)
	(position_embeddings): Embedding(512, 768)
	(token_type_embeddings): Embedding(2, 768)
	(LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
	(dropout): Dropout(p=0.1, inplace=False)
)

(encoder): BertEncoder(
	(layer): ModuleList(
		(0): BertLayer(
		(attention): BertAttention(
			(self): BertSelfAttention(
			(query): Linear(in_features=768, out_features=768, bias=True)
			(key): Linear(in_features=768, out_features=768, bias=True)
			(value): Linear(in_features=768, out_features=768, bias=True)
			(dropout): Dropout(p=0.1, inplace=False)
			)
			(output): BertSelfOutput(
			(dense): Linear(in_features=768, out_features=768, bias=True)
			(LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
			(dropout): Dropout(p=0.1, inplace=False)
			)
		)
		(intermediate): BertIntermediate(
			(dense): Linear(in_features=768, out_features=3072, bias=True)
			(intermediate_act_fn): GELUActivation()
		)
		(output): BertOutput(
			(dense): Linear(in_features=3072, out_features=768, bias=True)
			(LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
			(dropout): Dropout(p=0.1, inplace=False)
		)
		)
 		...
 )

(pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
)
# Pooling = 여러 값을 요약 = 
'''