# from utils import load_data
import os
from tokenizers import pre_tokenizers, ByteLevelBPETokenizer
from utils import load_data
VOCAB_SIZE = 512

    # Train tokenizer on your dataset
def batch_iterator(batch_size=1000, split='train', columns='abc'):
    for i in range(0,len(dataset[split]), batch_size):
        yield dataset[split][i : i + batch_size][columns]

if __name__ == '__main__':
    # load data
    dataset = load_data('melodyhub_dataset_standardized')
    
    # train tokenizer using transformers tokenizer library
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train_from_iterator(batch_iterator(),
                                  length=len(dataset['train']),
                                  special_tokens=["<START>", "<END>", "<PAD>"],
                                  vocab_size=VOCAB_SIZE)
    
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    tokenizer.save_model(".", "Tokenizer/ABCNotationTokenizer")

