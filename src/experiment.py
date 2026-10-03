import os
import sys

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader, Subset

from src.dataset import make_loaders
from src.evaluate import predict_loader, summarize
from src.model import CNNBaseline
from src.sampler import BalancedBatchSampler
from src.train import fit, make_scheduler
from src.utils import set_seed


def save_result(row, results_path):
    new = pd.DataFrame([row])
    if os.path.exists(results_path):
        try:
            old = pd.read_csv(results_path)
            if not old.empty and 'experiment' in old.columns:
                old = old[old['experiment'] != row['experiment']]
            new = pd.concat([old, new], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    new.to_csv(results_path, index=False)


def run_experiment(name, device, paths, epochs=20, img_size=128, batch_size=32,
                   augment=False, dropout_p=0.0, pool_type='max',
                   weight_decay=0.0, scheduler_name='none', lr=1e-3, seed=42,
                   train_subset=None, sampler_type='none', class_weights=False):
    set_seed(seed)

    train_ds, val_ds, _, val_loader = make_loaders(
        paths['data'], img_size, batch_size, seed, augment)

    if train_subset is not None:
        targets = [train_ds.targets[i] for i in train_subset]
        train_ds_use = Subset(train_ds, train_subset)
    else:
        targets = train_ds.targets
        train_ds_use = train_ds

    if sampler_type == 'balanced':
        n = len(train_ds_use)
        num_batches = (n + batch_size - 1) // batch_size
        sampler = BalancedBatchSampler(
            targets=targets, batch_size=batch_size,
            num_batches=num_batches, seed=seed)
        train_loader = DataLoader(train_ds_use, batch_sampler=sampler)
    else:
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds_use, batch_size=batch_size,
                                   shuffle=True, generator=g, num_workers=0)

    if class_weights:
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight
        w = compute_class_weight('balanced',
                                  classes=np.unique(targets), y=targets)
        w = torch.tensor(w, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    config = {'augment': augment, 'dropout_p': dropout_p, 'pool_type': pool_type,
              'weight_decay': weight_decay, 'scheduler': scheduler_name, 'lr': lr,
              'epochs': epochs, 'batch_size': batch_size, 'optimizer': 'AdamW',
              'sampler': sampler_type, 'class_weights': class_weights,
              'n_train': len(train_ds_use)}

    meta = {'architecture': 'CNNBaseline', 'class_to_idx': train_ds.class_to_idx,
            'img_size': img_size, 'seed': seed, 'config': config}
    model = CNNBaseline(dropout_p=dropout_p, pool_type=pool_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler, step_on_val = make_scheduler(scheduler_name, optimizer)

    os.makedirs(paths['ckpt_dir'], exist_ok=True)
    os.makedirs(paths['log_dir'], exist_ok=True)
    ckpt = os.path.join(paths['ckpt_dir'], f'{name}.pt')
    log = os.path.join(paths['log_dir'], f'{name}.csv')

    history = fit(model, train_loader, val_loader, optimizer, criterion, device,
                  epochs, ckpt, log, meta=meta,
                  scheduler=scheduler, step_on_val=step_on_val)

    model.load_state_dict(torch.load(ckpt, weights_only=False)['model_state_dict'])
    trues, preds = predict_loader(model, val_loader, device)
    row = summarize(trues, preds, train_ds.classes)
    best = max(history, key=lambda r: r['val_macro_f1'])
    row.update({'experiment': name, 'best_epoch': best['epoch'],
                'train_loss': best['train_loss'], 'val_loss': best['val_loss'],
                'gap': best['val_loss'] - best['train_loss'],
                'final_lr': history[-1]['lr'], **config})
    save_result(row, paths['results'])
    return row   