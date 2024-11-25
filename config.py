import torch as th

class GPT2Config:
    def __init__(self):
        self.embed_size = 768
        self.num_heads = 12
        self.head_size = 64
        self.num_layers = 12
        self.vocab_size = 256
        self.max_len = 1024
        self.block_size = 32
        self.lr = 3e-4
        self.batch_size = 82
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.path = 'models/model.pth'
        self.num_epochs = 100
        self.dataset = 'melodyhub_dataset'
        self.columns = 'dataset'
        
