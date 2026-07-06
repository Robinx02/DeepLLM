from datasets import load_dataset
import os

dataset = load_dataset("Josephgflowers/Finance-Instruct-500k" ,split = "train" ,streaming = True)
target_size_bytes = 50 * 1024 *1024
text = ""

for row in dataset:
    formatted = f"<|user|>\n{row['input']}\n<|assistant|>\n{row['output']}<|endoftext|>\n"
    text += formatted
    if len(text.encode("utf-8")) >= target_size_bytes:
        break

with open(r"C:\ptoh\DeepLLM\data\economy_train.txt" , "w" , encoding="utf-8") as f:
    f.write(text)

print(f"Saved {len(text.encode('utf-8'))/1024/1024:.2f} MB")