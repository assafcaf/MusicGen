import os
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor 


def encode_batch(batch, tokenizer, block_size, columns):
    """
    Encodes a batch of examples into fixed-size blocks.
    Args:
        batch (dict): A batch of examples.
        tokenizer (Tokenizer): The trained tokenizer.
        block_size (int): The fixed block size for encoding.
    Returns:
        dict: Encoded input IDs split into blocks.
    """
    # Loop through each text in the batch
    xs = []
    for text in batch[columns]:
        # Encode the text
        if len(text) > block_size//2:
            encoded = tokenizer.encode(text)

            input_ids = encoded.ids
            # Split into chunks of block_size
            x = [input_ids[i : i + block_size+1] for i in range(0, len(input_ids)-1, block_size+1)]
            # Pad the last block if needed
            for i in range(len(x)):
                if len(x[i]) < block_size+1:
                    x[i] += [tokenizer.token_to_id("<PAD>")] * (block_size - len(x[i])+1)
            xs.append(torch.tensor(x, dtype=int))

        
    # Append the blocks for this text to the result
    return torch.vstack(xs)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]

class BasicDataLoader:
    def __init__(self, ds, batch_size, tokenizer, shuffle=True, block_size=1024, device='cpu', columns='abc notation'): 
        self.ds = ds
        self.splits = list(ds.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = None
        self.tokenizer = tokenizer
        self.dataloaders = None
        self.block_size = block_size
        self.device = device
        self.columns = columns
        
    def get_batch(self):
        pass
    
    def set_split(self, split=None):
        self.split = split
    
    def encode_data(self):
        pass

class ABCNotationDataLoader(BasicDataLoader):
    def __init__(self, ds, batch_size, tokenizer, shuffle=True, block_size=1024, device='cpu', columns='abc notation'):
        super().__init__(ds, batch_size, tokenizer, shuffle, block_size, device, columns)
    
    def encode_data(self, splits=['train', 'validation'], columns='abc notation', batch_size=5000):
        print("Encoding data...")
        self.dataloaders = {}
        for split in self.ds:
            if split in splits:
                print(f"Encoding {split} data...")
                encoded_dataset = [self._encode_batch(self.ds[split][i:i+batch_size]) 
                                   for i in tqdm(range(0, len(self.ds[split]), batch_size),
                                                 desc=f"Encoding {split} data")]
                dataset = CustomDataset(torch.vstack(encoded_dataset).contiguous().to('cpu'))
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                self.dataloaders[split] = dataloader
        print()
    
    def get_batch(self, split=None):
        assert split in self.splits, f"Split {split} not found in dataset splits: {self.splits}"
        xy = next(iter(self.dataloaders[split]))
        return xy[0].to(self.device), xy[1].to(self.device)

    def process_split(self, split, batch_size, num_processes, columns='abc notation', percentage=1.):
        print(f"Encoding {split} data...")
        # Prepare batches
        p = max(int(len(range(0, len(self.ds[split]), batch_size) )* percentage), 1)
        batches = [self.ds[split][i:i + batch_size] for i in range(0, len(self.ds[split]), batch_size)[:p]]

        encoded_batches = []

        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(encode_batch, batch, self.tokenizer, self.block_size, columns)
                for batch in batches
            ]

            for future in tqdm(futures, desc=f"Encoding {split} data"):
                encoded_batches.append(future.result())

            # Combine encoded batches into a dataset
            dataset = CustomDataset(torch.vstack(encoded_batches).to('cpu').contiguous())
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def encode_data_parallel(self, splits=['train', 'validation'], num_processes=4, columns='abc notation', batch_size=10000, percentage=1.):
        print("Encoding data parallel...")
        self.dataloaders = {}
        for split in self.ds:
            if split in splits:
                self.dataloaders[split] = self.process_split(split, batch_size, num_processes, columns, percentage)


def load_data(data_root):
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    f_names = os.listdir(data_root)
    shareds_train = []
    shards_val = [] 
    for fname in f_names:
        print(f"reading {fname}...")
        with open(os.path.join(data_root, fname), "rb") as f:
            tokens = torch.tensor(np.frombuffer(f.read(), dtype=dtype)).long().contiguous()
        if fname.startswith('train'):
            shareds_train.append(tokens)
        else:
            shards_val.append(tokens)
    return shareds_train, shards_val


class ABCTokenizedDataset(Dataset):
    def __init__(self, data_root, block_size, device='cpu', dtype=torch.long, batch_size=1, max_batches_per_shard=100): 
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.shareds_train, self.shards_val = load_data(data_root)
        self.current_shard = 1
        self.batch_cout = 0
        self.split = 'train'
        self.max_batches_per_shard = 100
        
    def set_split(self, split):
        if split == 'train':
            self.split = split
            
    def __len__(self):
        return len(self.data) - self.block_size

    def handle_shard(self):
        if self.split == 'train':
            data = self.shareds_train[self.current_shard]
            self.batch_cout += 1
        else:
            data = self.shards_val[0]
        if self.batch_cout > self.max_batches_per_shard:
            self.current_shard = torch.randint(0, len(self.shareds_train), (1,)).item()
        return data
    
    def __getitem__(self, idx):
        data = self.handle_shard()
        try:
            idxs = torch.randint(0, len(data) - self.block_size, (self.batch_size,))
            d = torch.zeros(self.batch_size, self.block_size+1, dtype=self.dtype)
            for i, idx in enumerate(idxs):
                d[i] = data[idx : idx + self.block_size+1]
        except:
            x=1
        return d[:, :self.block_size].to(self.device), d[:, 1:].to(self.device)
