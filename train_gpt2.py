
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import load_data
from config import GPT2Config

from my_transformers.tokenizer import BPETokenizer, BPETransformers
from my_transformers.transformers import GPT
from my_transformers.pipelines import pipeline
from my_transformers.data_loader import ABCNotationDataLoader



with torch.no_grad():
    def estimate_loss(model, eval_iters, data_loader):
        out = {}
        model.eval()
        for split in dataset:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                xy = next(data_loader.get_batch("train"))
                x, y = xy[0], xy[1]
                _, loss = model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

if __name__ == '__main__':
    con = GPT2Config()
    # unique run name for saving models using time stamp
    run_name = time.strftime("models/%Y-%m-%d-%H-%M-%S.pth")
    run_pth = os.path.join("models/", run_name)
    os.makedirs(os.path.join("models/", run_name), exist_ok=True)    
    print(f"Device: {con.device}", end='\n\n')
    dataset = load_data(con.dataset)
    tokenizer = BPETokenizer(dataset, vocab_size=con.vocab_size, split='train', columns=con.columns)    
    data_loader = ABCNotationDataLoader(ds=dataset,
                                        batch_size=con.batch_size,
                                        tokenizer=tokenizer,
                                        shuffle=True,
                                        block_size=con.block_size,
                                        device=con.device,
                                        columns=con.columns)
    
    data_loader.encode_data(splits=dataset.keys())

    
    # Create a DataLoader for the training and validation data
    model = GPT(con)
    model.to(con.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=con.lr)

    # training loop
    n_iters = 5000
    print("Training begins...")
    with tqdm(total=n_iters, desc="Training Iterations") as pbar:
        for step in range(n_iters):
            # Start timing
            start_time = time.time()

            # Measure time to get the batch
            batch_start = time.time()
            xy = next(data_loader.get_batch("train"))
            x, y = xy[0], xy[1]
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
            if step % (n_iters // 10) == 0:
                losses = estimate_loss(model, 25, data_loader=data_loader)
            pbar.set_postfix(
                batch=f"{batch_time:.2f}s",
                forward=f"{forward_time:.2f}s",
                backward=f"{backward_time:.2f}s",
                total=f"{total_time:.2f}s",
                t_loss=losses['train'].item(),
                v_loss=losses['validation'].item()
                )
            pbar.update(1)
    print("Training complete!")
    # Save the model
    torch.save(model.state_dict(), os.path.join(run_pth, 'model.pth'))
    print("Model saved!")
    
    generator = pipeline('text-generation', model, tokenizer)
    print(generator('X:1\nT:Title\nK:Am\n', max_len=64)[0])

