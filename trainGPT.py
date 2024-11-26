
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader  

from utils import load_data, estimate_loss
from config import GPT2Config

from my_transformers.tokenizer import BPETokenizer, BPETransformers
from my_transformers.transformers import GPT
from my_transformers.pipelines import pipeline
from my_transformers.data_loader import ABCNotationDataLoader
from tokenizers import Tokenizer

if __name__ == '__main__':
    con = GPT2Config()
    # unique run name for saving models using time stamp
    run_dir = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_pth = os.path.join("models/", run_dir) 
    print(f"Device: {con.device}", end='\n\n')
    dataset = load_data(con.dataset)
    
    # Tokenize the dataset
    tokenizer = BPETokenizer(dataset, vocab_size=con.vocab_size, split='validation', columns=con.columns)       
    # encode the dataset 
    data_loader = ABCNotationDataLoader(ds=dataset,
                                        batch_size=con.batch_size,
                                        tokenizer=tokenizer,
                                        shuffle=True,
                                        block_size=con.block_size,
                                        device=con.device,
                                        columns=con.columns)
    
    data_loader.encode_data_parallel(splits=dataset.keys(), num_processes=4, percentage=.1)
    torch.set_float32_matmul_precision('medium')
    
    # Create a DataLoader for the training and validation data
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
            x, y = data_loader.get_batch("train")
            batch_time = time.time() - batch_start

            # Measure time for forward pass
            forward_start = time.time()
            optimizer.zero_grad()
            with torch.autocast(device_type=con.device, dtype=torch.float16):
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
                losses = estimate_loss(model, 1, splits=list(dataset.keys()), data_loader=data_loader)
                torch.save(model.state_dict(), os.path.join(run_pth, 'model.pth'))
            pbar.set_postfix(
                batch=f"{batch_time:.2f}s",
                backward=f"{backward_time:.2f}s",
                total=f"{total_time:.2f}s",
                t_loss=f"{losses['train'].item():.2f}",
                v_loss=f"{losses['validation'].item():.2f}"
                )
            pbar.update(1)
    print("Training complete!")
    # Save the model
    torch.save(model.state_dict(), os.path.join(run_pth, 'model.pth'))
    print("Model saved!")
    
    generator = pipeline('text-generation', model, tokenizer)
    print(generator('X:1\nT:Title\nK:Am\n', max_len=64)[0])

