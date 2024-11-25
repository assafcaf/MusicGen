
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast
def char_level_tokenizer(dataset):
    chars = sorted(set("\n\n".join(dataset["train"]["abc notation"]+dataset["validation"]["abc notation"])))
    vocab_size = len(chars) 
    print(f"vocab_size: {vocab_size}")
    print(f"chars: {chars}")
    chat2index = {ch:i for i, ch in enumerate(chars)}
    index2chat = {i:ch for i, ch in enumerate(chars)}
    encode = lambda x: [chat2index[c] for c in x]
    decode = lambda x: "".join([index2chat[c] for c in x])
    return encode, decode, vocab_size

def BPETokenizer(dataset, vocab_size=256, split=None, columns='abc notation', pth=None):
    assert split in ['train', 'test', 'validation', None], 'Split must be one of train, test, validation, or None'
    
    print("Training tokenizer...")
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Define pre-tokenization rules (split on |, :, and whitespace)
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="\n+", behavior="removed")
    # Train tokenizer on your dataset
    def batch_iterator(batch_size=1000):
        for i in range(0,len(dataset[split]), batch_size):
            yield dataset[split][i : i + batch_size][columns]
            
    trainer = trainers.BpeTrainer(special_tokens=["<START>", "<END>", "<PAD>"], vocab_size=vocab_size)
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset[split]))

    return tokenizer

def BPETransformers(dataset, vocab_size=256, split=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = tokenizer.train_new_from_iterator(dataset, vocab_size=vocab_size, show_progress=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer