from torch.utils.data import Dataset
import numpy as np
import torch
import os


def load_data(pth):
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    data = np.memmap(pth, dtype=dtype, mode='r')
    return data


class ABCNotationDataset(Dataset):
    def __init__(self, root_data, block_size, batch_size, split="train", device='cpu', dtype=torch.uint16): 
        assert split in ["train", "validation", "test"], "Split must be one of train, validation, or test"
        self.block_size = block_size
        self.batch_size = batch_size
        self.split = split
        self.device = device
        self.dtype = dtype
        
        self.data = {}
        f_names = os.listdir(root_data)
        for f in f_names:
            print(f"reading {f}...")
            self.data[f.split(".")[0]] = load_data(os.path.join(root_data, f))
        stop=1
    def set_split(self, split):
        self.split = split

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        d = torch.tensor(self.data[self.split][idx : idx + self.block_size+1], dtype=self.dtype)
        return d[:self.block_size].to(self.device), d[1:].to(self.device)

# example usage
if __name__ == '__main__':
    root_dir = r"/home/assaf_caftory/MusicGen/melodyhub_splited"
    block_size = 256
    batch_size = 16
    dl = ABCNotationDataset(root_dir, block_size=block_size, batch_size=batch_size, split="train")

    for x, y in dl:
        print(f"shapes: {x.shape}, {y.shape}")
        print(f"values\n\t{x}\n\t{y}")
        break