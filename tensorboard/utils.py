import numpy as np

def split_data(chars, batch_size, num_steps, split_frac=0.9):
    slice_size = batch_size * num_steps
    n_batches = int(len(chars) / slice_size)
    x = chars[: n_batches*slice_size]
    y = chars[1: n_batches*slice_size + 1]
    split_idx = int(batch_size*split_frac)
    
    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))

    split_idx = int(n_batches*split_frac)
    train_x, train_y= x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
    val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]
    return train_x, train_y, val_x, val_y

def get_batch(arrs, num_steps):
    batch_size, slice_size = arrs[0].shape
    
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]