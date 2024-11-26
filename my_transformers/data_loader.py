import torch
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
