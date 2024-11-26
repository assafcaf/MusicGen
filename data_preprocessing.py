# from utils import load_data
from tqdm import tqdm
from datasets import load_dataset


def preprocess(example):
    """Preprocess data to remove unwanted columns and add new"""
    
    col = 'input' if example['task'] != 'generation' else 'output'
    example['abc notation'] =  "<START>" + "\n".join(example[col].split("\n")[2:]) +"<END>"
    
    return example

if __name__ == '__main__':
    # load data
    dataset = load_dataset("sander-wood/melodyhub")
    
    x = set(dataset['train']["input"])
    # preprocess data
    updated_dataset = dataset.map(preprocess, batched=False, remove_columns=['input', 'dataset', 'task', 'output'], keep_in_memory=True)

    #  save cleaned data
    updated_dataset.save_to_disk('melodyhub_dataset_cleaned')
