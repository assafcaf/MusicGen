from utils import load_data
from tqdm import tqdm

dataset = load_data('melodyhub_dataset')

def add_entry(example):
    # Compute or set a new field, e.g., length of abc_notation
    col = 'input' if example['task'] != 'generation' else 'output'
    example['abc notation'] =  "<START>" + "\n".join(example[col].split("\n")[2:]) +"<END>"
    
    return example


updated_dataset = dataset.map(add_entry, batched=False, remove_columns=['input', 'dataset', 'task', 'output'], keep_in_memory=True)

            

#  save cleaned data
updated_dataset.save_to_disk('melodyhub_dataset_cleaned')
