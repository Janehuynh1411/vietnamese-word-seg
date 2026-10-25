import torch
from transformers import AutoModel, AutoTokenizer

# 1. Load the tokenizer and model (vinai/phobert-base or vinai/phobert-base-v2)
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
phobert = AutoModel.from_pretrained(model_name)

# 2. Input text must be word-segmented (syllables of a word connected with an underscore)
sentence = "Tôi là sinh_viên trường đại_học Công_nghệ ."

# 3. Encode the sentence to input IDs tensors
input_ids = torch.tensor([tokenizer.encode(sentence)])

# 4. Extract features without gradient calculation
with torch.no_grad():
    outputs = phobert(input_ids)
    # Last hidden states are typically the first element in the output tuple
    last_hidden_state = outputs.last_hidden_state

print("Input IDs:", input_ids)
print("Features shape:", last_hidden_state.shape)
