
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader  

from utils import load_data, estimate_loss
from config import GPT2Config
from my_transformers.transformers import GPT
from my_transformers.pipelines import pipeline
from my_transformers.data_loader import ABCTokenizedDataset


from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

if __name__ == '__main__':
    # set cude device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    con = GPT2Config()
    torch.set_float32_matmul_precision('medium')
    
    # unique run name for saving models using time stamp
    run_dir = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_pth = os.path.join("models/", run_dir) 
    print(f"Device: {con.device}", end='\n\n')
    
    # Tokenize the dataset
    tokenizer = tokenizer = ByteLevelBPETokenizer(
            vocab="/home/assaf_caftory/MusicGen/Tokenizer/ByteLevelBPETokenizer-512/ABCNotationTokenizer-vocab.json",
            merges="/home/assaf_caftory/MusicGen/Tokenizer/ByteLevelBPETokenizer-512/ABCNotationTokenizer-merges.txt",
        )

    dataloader = ABCTokenizedDataset(data_root=con.root_data, batch_size=con.batch_size, block_size=con.block_size, device=con.device, dtype=torch.long, max_batches_per_shard=con.max_batches_per_shard)
    
    # tdl = DataLoader(t_dset, batch_size=con.batch_size, shuffle=True)
    # vdl = DataLoader(v_dest, batch_size=con.batch_size, shuffle=True)
    #####################################################################################################################
    
    
    model = GPT(con)
    model.to(con.device)
    print("compiling model....")
    model = torch.compile(model)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=con.lr, betas=(0.9, 0.95), eps=1e-8)


    # training loop
    os.makedirs(os.path.join("models/", run_dir), exist_ok=True)   
    print("Training begins...")
    with tqdm(total=con.n_iters, desc="Training Iterations") as pbar:
        for step in range(con.n_iters):
            dataloader.set_split('train')
            # Start timing
            start_time = time.time()

            # Measure time to get the batch
            batch_start = time.time()
            x, y = next(iter(dataloader))
            batch_time = time.time() - batch_start

            # Measure time for forward pass
            forward_start = time.time()
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=con.device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            forward_time = time.time() - forward_start

            # Measure time for backward pass
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start
            
            # Measure token per second
            
            # Total step time
            total_time = time.time() - start_time

            # Print timing information
            # Estimate loss at intervals
            if step % (1000) == 0:
                losses = estimate_loss(model, 25, splits=['train', 'validation'], dataloader=dataloader)   
                checkpoint = {
                    'model': model.state_dict(),
                }
                torch.save(model.state_dict(), os.path.join(run_pth, 'model.pth'))
            pbar.set_postfix(
                batch=f"{batch_time:.2f}s",
                backward=f"{backward_time:.2f}s",
                total=f"{total_time:.2f}s",
                tokens=f"{con.batch_size * con.block_size / total_time:.2f}/s",
                t_loss=f"{losses['train'].item():.2f}",
                v_loss=f"{losses['validation'].item():.2f}"
                )
            pbar.update(1)
    print("Training complete!")
    # Save the model
    torch.save(model.state_dict(), os.path.join(run_pth, 'model.pth'))
    print("Model saved!")
