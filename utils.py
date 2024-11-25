import datasets


def load_data(pth, split=None):
    assert split in ['train', 'test', 'validation', None], 'Split must be one of train, test, validation, or None'
    ds = datasets.load_from_disk(pth)

    if split is None:
        return ds
    else:
        return ds[split]