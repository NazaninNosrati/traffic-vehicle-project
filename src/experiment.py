import os
import sys

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader, Subset

from src.dataset import make_loaders  , simulate_imbalance
from src.evaluate import predict_loader, summarize
from src.model import CNNBaseline
from src.sampler import BalancedBatchSampler
from src.train import fit, make_scheduler  , make_criterion 
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
                   imbalance=None, balanced=False , loss = 'ce'):
    set_seed(seed)
    train_ds, _, _, val_loader = make_loaders(
        paths['data'], img_size, batch_size, seed, augment)

    ds, targets = train_ds, train_ds.targets
    if imbalance is not None:
        keep = simulate_imbalance(train_ds.targets, imbalance, seed)
        ds = Subset(train_ds, keep)
        targets = [train_ds.targets[i] for i in keep]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=0)
    if balanced:
        sampler = BalancedBatchSampler(targets, batch_size,
                                       num_batches=len(train_loader), seed=seed)
        train_loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    model = CNNBaseline(dropout_p=dropout_p, pool_type=pool_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler, step_on_val = make_scheduler(scheduler_name, optimizer)
    criterion = make_criterion(loss , len(train_ds.classes))

    config = dict(augment=augment, dropout_p=dropout_p, pool_type=pool_type,
                  weight_decay=weight_decay, scheduler=scheduler_name, lr=lr,
                  epochs=epochs, batch_size=batch_size, optimizer='AdamW',
                  imbalance=str(imbalance), balanced=balanced , loss=loss)
    meta = {'architecture': 'CNNBaseline', 'class_to_idx': train_ds.class_to_idx,
            'img_size': img_size, 'seed': seed, 'config': config}
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