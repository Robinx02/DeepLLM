import torch
from torch.nn import functional as F

from model import TransformerModel , block_size , device
from BPEToken import BPE

#--- paths ---

weight_path = r'C:\ptoh\DeepLLM\output\checkpoint.pth'
tokenzier_path = r'C:\ptoh\DeepLLM\output\bpe_finance.json'
vocab_size = 8000

# --- load tokenizer --- 

bpe_token = BPE.load(tokenzier_path)

# ---load model---
model = TransformerModel(vocab_size=vocab_size).to(device)
