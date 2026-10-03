import os

import pandas as pd
import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * images.size(0)
        n += images.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(targets.cpu().tolist())
    return total_loss / n, f1_score(trues, preds, average='macro')


def make_scheduler(name, optimizer):
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5), False
    if name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3), True
    return None, False


def fit(model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs, ckpt_path, log_path, meta=None,
        scheduler=None, step_on_val=False):
    best_f1, history = -1.0, []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
               'val_macro_f1': val_f1, 'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        print(row)

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val_macro_f1': val_f1, **(meta or {})}, ckpt_path)

        if scheduler is not None:
            if step_on_val:
                scheduler.step(val_loss)
            else:
                scheduler.step()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pd.DataFrame(history).to_csv(log_path, index=False)
    return history