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


def split_data(dataset):
        # split data to files
    os.makedirs('melodyhub_splited', exist_ok=True)
    for split, dset in dataset.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join("melodyhub_splited", f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
def standardized_data(dataset):
    def process(example):
        col = task_map[example['task']]
        if col == 'skip':
            return None
        example['abc'] = "\n".join(example[col].split("\n")[2:])
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

def tokenize_data(dataset):
    def tokenize(example):
        ids = tokenizer.encode(example['abc']).ids # encode_ordinary ignores any special tokens
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out 

        # loaad tokenizer
    
    tokenizer = ByteLevelBPETokenizer(
            vocab="/home/assaf_caftory/MusicGen/Tokenizer/ABCNotationTokenizer-vocab.json",
            merges="/home/assaf_caftory/MusicGen/Tokenizer/ABCNotationTokenizer-merges.txt",
        )
    tokenizer._tokenizer.post_processor = BertProcessing(
    ("<START>", tokenizer.token_to_id("<END>")),
    ("<ENC>", tokenizer.token_to_id("<START>")),
    )
    
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
    
    # tokenize data
    if not os.path.exists('melodyhub_dataset_tokenized'):
        dataset = tokenize_data(dataset)
    else:
        dataset = load_data('melodyhub_dataset_tokenized')
    
    split_data(dataset)