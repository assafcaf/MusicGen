
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader  

from utils import load_data, estimate_loss
from config import GPT2Config


from my_transformers.transformers import GPT
from my_transformers.pipelines import pipeline
from my_transformers.data_loader import ABCNotationDataLoader
from transformers import DataCollatorWithPadding

from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


from torch.utils.data import Dataset, DataLoader

class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=8):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.data = self._process_texts()

    def _process_texts(self):
        processed_data = []
        for text in tqdm(self.texts[:10]):
            # Tokenize the text with overflowing tokens
            tokens = self.tokenizer(
                text,
                padding=True,  # Do not pad here
                max_length=self.max_length + 1,  # Include one extra token for labels
                truncation=True,
                return_overflowing_tokens=True,
                stride=0,
                return_tensors="pt"
            )
            # Flatten overflowing chunks
            input_ids = tokens["input_ids"]
            for chunk in input_ids:
                processed_data.append({
                    "input_ids": chunk[:-1],  # Input for the model
                    "labels": chunk[1:],    # Labels for the next-token prediction
                })
        return torch.vstack(processed_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def encode_batch(text, tokenizer, columns='abc notation'):
    
    # Encode the text
    tmp = tokenizer.encode(text[columns])
    return {'input_ids': tmp.ids}

if __name__ == '__main__':
    con = GPT2Config()
    # unique run name for saving models using time stamp
    run_dir = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_pth = os.path.join("models/", run_dir) 
    print(f"Device: {con.device}", end='\n\n')
    dataset = load_data(con.dataset)
    
    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained("tokenizers", fast=True)
    con.vocab_size = tokenizer.vocab_size
    # encode the dataset 
    texts = dataset["train"][con.columns]  # Assuming con.columns points to the correct column
    tokenized_dataset = TokenizedDataset(texts, tokenizer, max_length=con.block_size)
    
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True)
    dataloader = DataLoader(dataset=tokenized_dataset, collate_fn=collate_fn, batch_size=con.batch_size)
    #####################################################################################################################
    
    
    model = GPT(con)
    model.to(con.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=con.lr)

    
    # training loop
    os.makedirs(os.path.join("models/", run_dir), exist_ok=True)   
    print("Training begins...")
    with tqdm(total=con.n_iters, desc="Training Iterations") as pbar:
        for step in range(con.n_iters):
            # Start timing
            start_time = time.time()

            # Measure time to get the batch
            batch_start = time.time()
            d = next(iter(dataloader))
            x, y = d['input_ids'].to(con.device), d['labels'].to(con.device)
            batch_time = time.time() - batch_start

            # Measure time for forward pass
            forward_start = time.time()
            optimizer.zero_grad()
            logits, loss = model(x, y)
            forward_time = time.time() - forward_start

            # Measure time for backward pass
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start

            # Total step time
            total_time = time.time() - start_time

            # Print timing information
            # Estimate loss at intervals
            if step % (1000) == 0:
                torch.save(model.state_dict(), os.path.join(run_pth, 'model.pth'))
            pbar.set_postfix(
                batch=f"{batch_time:.2f}s",
                backward=f"{backward_time:.2f}s",
                total=f"{total_time:.2f}s"
                )
            pbar.update(1)
    print("Training complete!")
    # Save the model
    torch.save(model.state_dict(), os.path.join(run_pth, 'model.pth'))
    print("Model saved!")
    
    generator = pipeline('text-generation', model, tokenizer)
    print(generator('X:1\nT:Title\nK:Am\n', max_len=64)[0])
    
