
import os
import torch
from utils import load_data
from config import GPT2Config

from my_transformers.transformers import GPT
from my_transformers.pipelines import pipeline

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    con = GPT2Config()
    print(f"Device: {con.device}", end='\n\n')
    
    run_pth = r"/home/assaf_caftory/MusicGen/runs/2024-12-01-12-33-26"
    tokenizer = tokenizer = ByteLevelBPETokenizer(
        vocab="/home/assaf_caftory/MusicGen/Tokenizer/ByteLevelBPETokenizer-512/ABCNotationTokenizer-vocab.json",
        merges="/home/assaf_caftory/MusicGen/Tokenizer/ByteLevelBPETokenizer-512/ABCNotationTokenizer-merges.txt",
    )

    # Create a DataLoader for the training and validation data
    model = GPT(con)
    checkpoint = torch.load(os.path.join(run_pth, 'step0_model.bin'), weights_only=True)
    model.set_model_wieghts(checkpoint)
    model.to(con.device)
    
    # Create a pipeline for text generation
    generator = pipeline('text-generation', model, tokenizer)
    prompt = input("Enter a prompt: ")
    while prompt != 'quit':
        print("Generated text: ")
        print(generator(prompt, max_len=256))
        prompt = input("Enter a prompt: ")
        print("")
