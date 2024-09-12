from .dataloader_cyclegan import load_data as load_cyclegan
from .dataloader_single import load_data as load_single

def load_data(dataname, batch_size, val_batch_size, root_smoke, root_clean, num_workers, img_size, **kwargs):
    if dataname == 'cycle':
        return load_cyclegan(batch_size, val_batch_size, root_smoke, root_clean, img_size)
    elif dataname == 'single':
        return load_single(batch_size, val_batch_size, root_smoke, img_size)
