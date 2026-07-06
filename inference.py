import torch
from torch.nn import functional as F

from model import TransformerModel , block_size , device
from BPEToken import BPE

#---paths---

weight_path = r'C:\ptoh\DeepLLM\output\checkpoint.pth'
tokenzier_path = r'C:\ptoh\DeepLLM\output\bpe_finance.json'
vocab_size = 8000

# ---load tokenizer--- 

bpe_token = BPE.load(tokenzier_path)

# ---load model---
model = TransformerModel(vocab_size=vocab_size).to(device)
checkpoint = torch.load(weight_path, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded from iter {checkpoint['iter']} | val loss {checkpoint['val_loss']:.4f}")

#--- inference ---

def chat(question , max_new_tokens=300 , temprature = 0.8 , top_k=50):
    prompt  = f"<|user|>\n{question}\n<|assistant|>\n"
    prompt_ids = bpe_token.encode(prompt)

    eos_ids = bpe_token.encode("<|endoftext|>")
    eos_id = eos_ids[0] if len(eos_ids) == 1 else None

    idx = torch.tensor([prompt_ids] , dtype = torch.long , device= device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,_ = model(idx_cond)
            logits = logits[:,-1,:] / temprature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits , dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_id and idx_next.item() == eos_id:
                break

            idx = torch.cat([idx, idx_next], dim=1)
    generated = idx[0, len(prompt_ids):].tolist()
    return bpe_token.decode(generated)

#__for running__

if __name__ == "__main__":
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit","quit"):
            break
        answer = chat(question)    
        print(f"\nAssistant: {answer}")    