import torch as th

class GPT2Config:
    def __init__(self):
        self.embed_size = 768
        self.num_heads = 12
        self.head_size = 64
        self.num_layers = 6
        self.vocab_size = 512
        self.max_len = 1024
        self.block_size = 512
        self.lr = 1e-4
        self.batch_size = 24
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.num_epochs = 100
        self.root_data = r"/home/assaf_caftory/MusicGen/melodyhub_splited"
        self.columns = 'abc notation'
        self.n_iters = 100000
        self.max_batches_per_shard = 100
    
    
