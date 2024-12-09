import datasets
import torch
import json
import math

def load_data(pth, split=None):
    assert split in ['train', 'test', 'validation', None], 'Split must be one of train, test, validation, or None'
    ds = datasets.load_from_disk(pth)

    if split is None:
        return ds
    else:
        return ds[split]


with torch.no_grad():
    def estimate_loss(model, eval_iters , splits, dataloader):
        
        out = {}
        model.eval()
        for split in splits:
            losses = torch.zeros(eval_iters)
            dataloader.set_split(split)
            for i in range(eval_iters):
                x, y = next(iter(dataloader))
                _, loss = model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

def save_config_to_json(config_instance, file_path):
    """
    Save the attributes of a configuration class instance to a JSON file.

    Args:
        config_instance (object): An instance of a class with configuration data.
        file_path (str): The path where the JSON file will be saved.

    Raises:
        TypeError: If the instance has non-serializable attributes.
    """
    # Extract attributes from the instance
    config_dict = {key: value for key, value in vars(config_instance).items()}
    
    # Save to a JSON file
    try:
        with open(file_path, 'w') as json_file:
            json.dump(config_dict, json_file, indent=4)
        print(f"Configuration saved to {file_path}")
    except TypeError as e:
        raise TypeError(f"Failed to serialize configuration data. {e}")

def get_scheduler(con):
    """
    Get the learning rate for the current step.

    Args:
        step (int): The current training step.

    Returns:
        float: The learning rate for the current step.
    """
    def get_lr(step):
        # Linear warmup followed by cosine decay
        if step < con.warmup_steps:
            return con.max_lr * (step+1) / con.warmup_steps
        
        # after max_steps, decay to min_lr
        if step >= con.max_steps:
            return con.min_lr
        
        # Cosine decay
        decay_rate = (step - con.warmup_steps) / (con.max_steps - con.warmup_steps)
        assert 0 <= decay_rate <= 1
        coef = 0.5 * (1.0 + math.cos(math.pi * decay_rate))
        return con.min_lr + coef*(con.max_lr - con.min_lr)
    return get_lr
    
        