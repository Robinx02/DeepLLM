# DeepLLM — GPT-Style Language Model from Scratch

From scratch implementation of a GPT style causal transformer trained on a financial Q&A dataset, built entirely in PyTorch without any high-level abstractions.

---

## Overview

This project implements a decoder only transformer language model trained on the [Josephgflowers/Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k) dataset from HuggingFace. The goal was to build a finance-domain Q&A bot by training a custom GPT architecture from scratch which included the tokenizer, training loop, and inference pipeline. Although the results were not that good due too the model simplicity and being trained on small dataset of 50mb.


## Project Structure

```
DeepLLM/
│
├── model.py          # Transformer architecture (Head, MHA, FFN, Block, TransformerModel)
├── train.py          # Training loop with checkpointing and loss curve saving
├── inference.py      # Interactive Q&A inference with temperature + top-k sampling
├── BPEToken.py       # Custom BPE tokenizer (fit, encode, decode, save, load)
│
├── data/
│   ├── dataload.py          # Streams and saves 50MB from HuggingFace dataset
│   ├── economy_train.txt    # Formatted Q&A training corpus (~50MB)
│   └── encoded_data.pt      # Pre-encoded token tensor (cached, not tracked by git)
│
└── output/
    ├── checkpoint.pth       # Best model checkpoint (not tracked by git)
    ├── bpe_finance.json     # Saved BPE tokenizer (not tracked by git)
    └── loss_curve.png       # Training and validation loss plot
```

---

## Key Implementation Details

### Custom BPE Tokenizer
Built a Byte-Pair Encoding tokenizer from scratch with:
- Base vocabulary of 256 UTF-8 bytes
- 7,744 learned merge operations
- Numpy-accelerated encoding via chunk-based `encode_batch()`
- JSON serialization for save/load across runs

### Attention Scaling Fix
Correctly scales dot-product attention by `head_size` (not `n_embd`):
```python
wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
```

### Training Pipeline
- **Optimizer:** AdamW (`lr=3e-4`)
- **Scheduler:** Cosine annealing to `1e-5`
- **Batch size:** 64
- **Iterations:** 20,000
- **Checkpoint:** Saved every 500 steps — resumes automatically if interrupted
- **Evaluation:** 200-batch rolling average on train and val splits

### Data Format
Each Q&A pair formatted with special tokens:
```
<|user|>
{question}
<|assistant|>
{answer}<|endoftext|>
```

---

## Training Results

|
 Metric 
|
 Value 
|

|
---
|
---
|

|
 Final train loss 
|
 ~1.6 
|

|
 Final val loss 
|
 ~1.35 
|

|
 Training tokens 
|
 17,525,704 
|

|
 Training time 
|
 ~6-8 hours (RTX 4060) 
|


> Note: Val loss lower than train loss is expected — dropout (0.2) is active during training but disabled during evaluation, making training loss appear artificially higher.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Robinx02/DeepLLM.git
cd DeepLLM

# Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows

# Install dependencies
pip install torch numpy matplotlib datasets
```

---

## Usage

### 1. Download and prepare dataset
```bash
python data/dataload.py
```
Streams ~50MB from HuggingFace Finance-Instruct-500k and saves as `economy_train.txt`.

### 2. Train the model
```bash
python train.py
```
- First run: fits BPE tokenizer on 500k chars, encodes full dataset, saves both
- Subsequent runs: loads pre-encoded `encoded_data.pt` instantly
- Checkpoints saved every 500 steps to `output/checkpoint.pth`
- Loss curve saved to `output/loss_curve.png` after training

### 3. Run inference
```bash
python inference.py
```
Interactive loop — type any finance question and get a generated answer:
```
You: What causes inflation?