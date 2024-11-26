
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


if __name__ == '__main__':
    con = GPT2Config()
    print(f"Device: {con.device}", end='\n\n')
    run_pth = r"/home/assaf_caftory/MusicGen/models/2024-11-26-16-22-46"
    dataset = load_data(con.dataset)
    tokenizer = BPETokenizer(dataset, vocab_size=con.vocab_size, split='validation', columns=con.columns)       




    # Create a DataLoader for the training and validation data
    model = GPT(con)
    checkpoint = torch.load(os.path.join(run_pth, 'model.pt'), weights_only=True)
    model.set_model_wieghts(checkpoint)
    model.to(con.device)
    
    # Create a pipeline for text generation
    generator = pipeline('text-generation', model, tokenizer)
    
    prompt = input("Enter a prompt: ")
    while prompt != 'quit':
        print("Generated text: ")
        print(generator(prompt, max_len=512)[0])
        prompt = input("Enter a prompt: ")
        print("")
