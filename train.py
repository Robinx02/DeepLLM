import torch
import os
from model import TransformerModel, block_size, device
from BPEToken import BPE
#-----Hyperparameters------

batch_size = 64
max_iters =5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
vocab_size = 3000
weight_path = r'C:\ptoh\DeepLLM\output'  #file where weights are saved

#--- Data Preparation ---
path = r'C:\ptoh\DeepLLM\data\input.txt'
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
else:
    text = ''
    print(f"No file found at {path}. Using empty fallback text.")

bpe_token = BPE(vocab_size)
bpe_token.fit(text[:10**5])  #training the tokenizer on the first 100k tokens

#splitting dataset into train and test split

data = torch.tensor(bpe_token.encode(text) , dtype=torch.long)
n = int(0.9*len(data))  #90% for training
train_data = data[:n]
val_data = data[n:]
