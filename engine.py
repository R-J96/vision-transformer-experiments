"""Train and eval functions
"""
import math
import sys
import torch
import torch.nn.functional as F
import numpy as np
from utils import calc_cls_measures


def train_one_epoch(model,
                    criterion,
                    data_loader,
                    optimiser,
                    device
                    ):
    y_probs = np.zeros((0, len(data_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []
    model.train()

    for i, (x, labels) in enumerate(data_loader):
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # zero parameter gradients
        optimiser.zero_grad(set_to_none=True)

        # compute forward, backward pass and optimise
        with torch.cuda.amp.autocast():
            outputs = model(x)
            loss = criterion(outputs, labels)

        loss.backward()
        optimiser.step()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        losses.append(loss_value)

        y_prob = F.softmax(outputs, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, labels.cpu().numpy()])

    train_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)

    return train_loss_epoch, metrics


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    y_probs = np.zeros((0, len(data_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    # switch to evaluation mode
    model.eval()

    for i, (x, labels) in enumerate(data_loader):
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(x)
            loss = criterion(outputs, labels)

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(outputs, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, labels.cpu().numpy()])

    val_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    return val_loss_epoch, metrics
