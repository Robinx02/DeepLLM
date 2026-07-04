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

def get_batch(split):
    #generating small batches of data of inpur x and y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size , (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size +1] for i in ix])
    x,y = x.to(device) , y.to(device)
    return x , y



@torch.no_grad()
def estimate_loss()
    
    out = {}
    model.eval()
    for split in ['train' , 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X , Y = get_batch(split)
            logits , loss= model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = TransformerModel(vocab_size = vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters() , lr = learning_rate)

start_iter = 0 # default starting iteration

# to check if checkpoints exist for model 

if os.path.exists(weight_path):
    print(f"weight exists at {weight_path}")
    checkpoint = torch.load(weight_path , map_location = device)

    #restoring the opetimizer and model states

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter'] + 1

    print(f"loaded the training checkpoint")

else:
    print("no previous saved weights")

for iter in range(max_iters):

    #every once in a while evalute the loss in train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"stop {iter}: train loss {losses['train']:.4f} , val loss {losses['val']:.4f}")

    xb , yb = get_batch('train')

    #sampling batch of data
    logits , loss = model(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()




