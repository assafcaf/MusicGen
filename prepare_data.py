# from utils import load_data
import os
from tqdm import tqdm
import numpy as np
from utils import load_data
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


task_map = {"cataloging": 'input',
            "generation": 'output',
            "harmonization": 'input',
            "melodization": 'input',
            "segmentation": 'input',
            "transcription": 'output'}

SYMBALS_TO_REMOVE = ["%", "T", "E:", "S:", "B:", "X:"]

def split_data(dataset, max_file_size_mb=40):
    """
    Splits the dataset into .bin files with a fixed maximum size by buffering tokens.

    Args:
        dataset (dict): A dictionary containing dataset splits (e.g., train, test, validation).
        max_file_size_mb (int): Maximum size of each .bin file in megabytes.
    """
    # Create output directory
    os.makedirs('melodyhub_splited', exist_ok=True)
    dtype = np.uint16  # Can be used since max_token_value < 2**16
    dtype_size = np.dtype(dtype).itemsize  # Size of one element in bytes

    for split, dset in dataset.items():
        buffer = []  # Initialize an empty buffer for tokens
        buffer_size_bytes = 0  # Track the buffer's size in bytes
        file_index = 0  # File index for naming

        for exmaple in tqdm(dset, desc=f"Spliting {split}"):
            # Get a shard of the dataset and concatenate token IDs
            tokens = exmaple['ids']
            sample_size_bytes = len(tokens) * dtype_size  # Size of the current sample in bytes
            buffer_size_bytes = len(buffer) * dtype_size 

            if buffer_size_bytes + sample_size_bytes < max_file_size_mb * 2**20:
                buffer.extend(tokens)
            else :
                filename = os.path.join("melodyhub_splited", f"{split}_{file_index}.bin")
                with open(filename, "wb") as f:
                    f.write(np.array(buffer, dtype=dtype).tobytes())
                buffer = tokens
                buffer_size_bytes = sample_size_bytes
                file_index += 1
        if len(buffer) > 0:
            filename = os.path.join("melodyhub_splited", f"{split}_{file_index}.bin")
            with open(filename, "wb") as f:
                f.write(np.array(buffer, dtype=dtype).tobytes())
            
    
def standardized_data(dataset):
    def process(example):
        col = task_map[example['task']]
        text = example[col]
        text =  "\n".join([s for s in text.split("\n")[:-1] if s[0] not in SYMBALS_TO_REMOVE]) 
        example['abc'] ="<s>" +  text + '</s>'
        return example
    dataset = dataset.filter(lambda example: example['task'] != 'variation', desc="Filtering variation tasks")

    dataset = dataset.map(
        process,
        remove_columns=['output', 'task', 'dataset', 'input'],
        desc="standardizing data",
        num_proc=num_proc,
    )
    dataset.save_to_disk('melodyhub_dataset_standardized')
    return dataset

def tokenize_data(dataset, tokenizer=None):
    def tokenize(example):
        ids = tokenizer.encode(example['abc']).ids # encode_ordinary ignores any special tokens
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out 
 
    # tokenize data
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=['abc'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    tokenized_dataset.save_to_disk('melodyhub_dataset_tokenized')
    return tokenized_dataset

if __name__ == '__main__':

    num_proc = 16
    
    # load data
    # if the standardized dataset doesn't exist, create it
    if not os.path.exists('melodyhub_dataset_standardized'):
        dataset = load_dataset("sander-wood/melodyhub")
        dataset = standardized_data(dataset)
    else:
        dataset = load_data('melodyhub_dataset_standardized')
    
    tokenizer = ByteLevelBPETokenizer(
        vocab="/home/assaf_caftory/MusicGen/Tokenizer/ByteLevelBPETokenizer-512/ABCNotationTokenizer-vocab.json",
        merges="/home/assaf_caftory/MusicGen/Tokenizer/ByteLevelBPETokenizer-512/ABCNotationTokenizer-merges.txt",
    )
    # tokenize data
    if not os.path.exists('melodyhub_dataset_tokenized'):
        dataset = tokenize_data(dataset, tokenizer=tokenizer)
    else:
        dataset = load_data('melodyhub_dataset_tokenized')
        
    if not os.path.exists('melodyhub_splited'):
        split_data(dataset)
    
    # sanity check
    tokens = dataset["validation"][0]['ids']
    d_dset = tokenizer.decode(tokens)
    print(d_dset)
    
    l = len(tokens)
    # load first split
    with open("melodyhub_splited/validation_0.bin", "rb") as f:
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    d_file = tokenizer.decode(tokens[:l])
    print(f"first sample:\n{d_file}")
    
    print(f" is equal: {(d_dset == d_file)}")
    
    