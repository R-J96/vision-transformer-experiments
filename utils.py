import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from functools import partial


def move_to(var, device):
    """Moves an input data collection to an input device

    Args:
        var (dict or list or tuple): data collection to be moved
        device (torch.device): device to move data to

    Returns:
        var: returns input data collection on new device
    """
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return [move_to(v, device) for v in var]
    elif isinstance(var, tuple):
        return (move_to(v, device) for v in var)
    return var.to(device)


def calc_cls_measures(probs, label):
    """Calculates classification metrics

    Args:
        probs (np.array): array of model's predicted probabilities for each class
        label (np.array): array of ground truth labels for each example

    Returns:
        dict: dictionary of classification metrics
    """
    label = label.reshape(-1, 1)
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(label, preds)
    f1 = f1_score(label, preds)
    recall = recall_score(label, preds)
    precision = precision_score(label, preds)
    auc = roc_auc_score(label, preds)

    metric_collects = {'accuracy': accuracy,
                       'f1 score': f1,
                       'recall': recall,
                       'precision': precision,
                       'auc roc': auc}
    return metric_collects


def save(*args, **kwargs):
    """Saves PyTorch model
    """
    torch.save(*args, **kwargs)


def to_tensor(x, dtype=torch.int32, device=None):
    """Return a tensor of input data collection x

    Args:
        x (): input data to convert
        dtype (torch.dtype, optional): Data type to convert x to within tensor. Defaults to torch.int32.
        device (torch.device, optional): Which device to store output tensor on. Defaults to None.

    Returns:
        torch.Tensor: Tensor of type dtype on device
    """
    device = torch.device('cpu') if device is None else device
    if torch.is_tensor(x):
        return x.to(device)

    return torch.tensor(x, dtype=dtype, device=device)


def to_dtype(x, dtype):
    """Cast Tensor x to the dtype """
    return x.type(dtype)


to_float16 = partial(to_dtype, dtype=torch.float16)
to_float32 = partial(to_dtype, dtype=torch.float32)
to_float64 = partial(to_dtype, dtype=torch.float64)
to_double = to_float64
to_int8 = partial(to_dtype, dtype=torch.int8)
to_int16 = partial(to_dtype, dtype=torch.int16)
to_int32 = partial(to_dtype, dtype=torch.int32)
to_int64 = partial(to_dtype, dtype=torch.int64)


def expand_many(x, axes):
    """Call expand_dims many times on x once for each item in axes."""
    for ax in axes:
        x = torch.unsqueeze(x, ax)
    return x
