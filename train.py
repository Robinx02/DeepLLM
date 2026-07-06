import torch
import tiktoken
import matplotlib.pyplot as plt
import os
from model import TransformerModel, block_size, device
#from BPEToken import BPE
#-----Hyperparameters------

batch_size = 128
max_iters =20000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
#vocab_size = 8000
weight_path = r'C:\ptoh\DeepLLM\output\checkpoint.pth'  #file where weights are saved

#--- Data Preparation ---
path = r'C:\ptoh\DeepLLM\data\economy_train.txt'
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
else:
    text = ''
    print(f"No file found at {path}. Using empty fallback text.")

#bpe_token = BPE(vocab_size)
#bpe_token.fit(text[:500_000])  #training on 500k chars
#bpe_token.save(r'C:\ptoh\DeepLLM\output\bpe_finance.json')

#this takes around 3-5hrs just for the BPE fitting alone because pure python implementation is O(n^2) merges get exponentially slower


enc = tiktoken.get_encoding("gpt2")
vocab_size = 50257

#splitting dataset into train and test split

#data = torch.tensor(bpe_token.encode(text) , dtype=torch.long)
print("Tokenizing text...")
data = torch.tensor(enc.encode(text,allowed_special = {"<|endoftext|>"}) ,dtype = torch.long)
print(f"Tokenization done — {len(data):,} tokens")
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
def estimate_loss():
    
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max = max_iters , eta_min=1e-5)
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

# Initialize lists to store losses
train_losses = []
val_losses = []
steps = []

for iter in range(start_iter , max_iters):

    #every once in a while evalute the loss in train and val sets
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"stop {iter}: train loss {losses['train']:.4f} , val loss {losses['val']:.4f}")
        #saving the current model state
        steps.append(iter)
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])

        checkpoint = {
            'iter':iter , 
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'val_loss': losses['val']
        }
        torch.save(checkpoint ,weight_path)
        print(f"weights and training checkpoint saved to {weight_path}")
    
    xb , yb = get_batch('train')
    
    #sampling batch of data
    logits , loss = model(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    scheduler.step()


#plotting the loss graph

plt.figure(figsize=(8, 6))
plt.plot(steps, train_losses, label='Train Loss')
plt.plot(steps, val_losses, label='Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Time')
plt.legend()
plt.savefig(r'C:\ptoh\DeepLLM\output\loss_curve.png')
print("saved the loss curve")


print('Training complete')

