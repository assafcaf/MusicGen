import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

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
        self.dataloaders = {}
        tokenizer, block_size = self.tokenizer, self.block_size
        def encode_batch(batch):
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
            for text in batch[self.columns]:
                # Encode the text
                encoded = tokenizer.encode(text)
                input_ids = encoded.ids
                # Split into chunks of block_size
                x = [input_ids[i : i + block_size+1] for i in range(0, len(input_ids)-1, block_size+1)]
                # Pad the last block if needed
                if len(x[-1]) < block_size+1:
                    x[-1] += [tokenizer.token_to_id("<PAD>")] * (block_size - len(x[-1])+1)
                xs.append(torch.tensor(x, dtype=int))
                
            # Append the blocks for this text to the result
            return torch.vstack(xs)
        

        print("Encoding data...")
        for split in self.ds:
            if split in splits:
                print(f"Encoding {split} data...")
                encoded_dataset = [encode_batch(self.ds[split][i:i+batch_size]) 
                                   for i in tqdm(range(0, len(self.ds[split]), batch_size),
                                                 desc=f"Encoding {split} data")]
                dataset = CustomDataset(torch.vstack(encoded_dataset).contiguous().to(self.device))
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                self.dataloaders[split] = dataloader
                
        print()
    
    def get_batch(self, split=None):
        assert split in self.splits, f"Split {split} not found in dataset splits: {self.splits}"
        yield next(iter(self.dataloaders[split] ))





    