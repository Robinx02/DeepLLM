import json
import numpy as np

class BPE:

    def __init__(self,vocab_size):
        self.vocab_size = vocab_size
        self.base_vocab = 256
        self.num_merges =  self.vocab_size - self.base_vocab
        self.merges = {}
        self.vocab = {}
        self.merge_rank = {}

    def get_pairs(self,ids):
        counts = {}
        for pair in zip(ids[0:], ids[1:]):
            counts[pair] = counts.get(pair,0) +1
        return counts

    def merge(self,ids , pair , idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids 

    def display_merges(self):
        for item in self.char_merges.items():
            if item is not None:
                ch1 , ch2 = item[0]
                print(f"'{ch1}'+'{ch2}'---> {item[1]}")

    def fit(self,text):
        merges = {}
        vocab = {idx:bytes([idx]) for idx in range(256)}  #int -> bytes
        ids = list(text.encode("utf-8"))
        
        for i in range(self.num_merges):
            pairs = self.get_pairs(ids)
            if not pairs:
                break
            top_pair = max(pairs , key = pairs.get)
            idx = 256 + i
            ids = self.merge(ids , top_pair , idx)

            #saving the merge
            merges[top_pair] = idx

            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if (i + 1) % 100 ==0:
                print(f"merge {i+1}/{self.num_merges}" , flush = True)
        self.merges = merges
        self.vocab = vocab
        self._merge_rank = {pair: rank for rank , pair in enumerate(merges.keys())}

    def encode(self,text):
        
        tokens = list(text.encode('utf-8'))
        if len(tokens) < 2:
            return tokens

        merge_rank = {pair: i for i, pair in enumerate(self.merges.keys())}

        while len(tokens) >= 2:
            arr = np.array(tokens, dtype=np.int32)

            pairs = list(zip(arr[:-1].tolist(), arr[1:].tolist()))
            
            best_rank = float('inf')
            best_pair = None
            for pair in pairs:
                rank = merge_rank.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None or best_rank == float('inf'):
                break

            idx = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens       

    def encode_batch(self, text, chunk_size=100_000):
        print(f"Encoding {len(text)/1_000_000:.1f}MB in chunks...", flush=True)
    
        chunks = text.split("<|endoftext|>")
        all_ids = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            ids = self.encode(chunk + "<|endoftext|>")
            all_ids.extend(ids)

            if (i + 1) % 100 == 0:
                print(f"  chunk {i+1}/{len(chunks)} — {len(all_ids):,} tokens so far", flush=True)

        print(f"Encoding done — {len(all_ids):,} total tokens", flush=True)
        return all_ids
    
    #decode a list of tokens back into string
    def decode(self,ids):
        
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8" , errors = "replace")
        return text 

    def save(self , filepath):
        data = {
            'vocab_size' : self.vocab_size,
            'merges' : {f"{k[0]}-{k[1]}": v for k , v in self.merges.items()},
            'vocab' : {k: list(v) for k, v in self.vocab.items()}
        }

        with open(filepath , 'w') as f:
            json.dump(data, f)

    # load the tokenizer state from a config file
    @classmethod
    def load(cls , filepath):
        with open(filepath,'r') as f:
            data = json.load(f)
            tokenizer = cls(data['vocab_size'])
            tokenizer.merges = {
                tuple(map(int, k.split('-'))) : v
                for k , v in data['merges'].items()
            }        
            tokenizer.vocab = {int(k): bytes(v) for k,v in data['vocab'].items()}
            tokenizer._merge_rank = {
                pair:i for i, pair in enumerate(tokenizer.merges.keys())
            }
            return tokenizer 
        
                
