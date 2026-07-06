from datasets import load_dataset

dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train", streaming=True)
target_size_bytes = 50 * 1024 * 1024

chunks = []
total_bytes = 0
count = 0

for row in dataset:
    formatted = f"<|user|>\n{row['user']}\n<|assistant|>\n{row['assistant']}<|endoftext|>\n"
    encoded = formatted.encode("utf-8")
    chunks.append(formatted)
    total_bytes += len(encoded)
    count += 1
    if count % 100 == 0:
        print(f"{count} rows — {total_bytes/1024/1024:.2f} MB")
    if total_bytes >= target_size_bytes:
        break

text = "".join(chunks)
with open(r"C:\ptoh\DeepLLM\data\economy_train.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Saved {total_bytes/1024/1024:.2f} MB — {count} rows")