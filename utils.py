import datasets
import torch

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