import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets

from src.dataset import get_transforms
from src.model import CNNBaseline


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds, trues = [], []
    for images, targets in loader:
        out = model(images.to(device))
        preds.extend(out.argmax(1).cpu().tolist())
        trues.extend(targets.tolist())
    return trues, preds

def summarize(trues, preds, class_names):
    p, r, f1, support = precision_recall_fscore_support(
        trues, preds, labels=list(range(len(class_names))),
        average=None, zero_division=0)
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        trues, preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        trues, preds, average='weighted', zero_division=0)
    acc = accuracy_score(trues, preds)
    
    row = {
        'macro_precision': p_macro,
        'macro_recall': r_macro,
        'macro_f1': f1_macro,
        'weighted_precision': p_weighted,
        'weighted_recall': r_weighted,
        'weighted_f1': f1_weighted,
        'accuracy': acc,
        'lowest_recall_class': class_names[int(r.argmin())],
        'lowest_precision_class': class_names[int(p.argmin())],
        'lowest_f1_class': class_names[int(f1.argmin())],
        'highest_f1_class': class_names[int(f1.argmax())],
    }
    
    for i, c in enumerate(class_names):
        row[f'recall_{c}'] = r[i]
        row[f'precision_{c}'] = p[i]
        row[f'f1_{c}'] = f1[i]
        row[f'support_{c}'] = int(support[i])
    
    return row

@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    logits, trues = [], []
    for images, targets in loader:
        logits.append(model(images.to(device)).cpu())
        trues.extend(targets.tolist())
    return torch.cat(logits), torch.tensor(trues)


@torch.no_grad()
def evaluate_on_test(model, test_root, device, class_names, 
                     img_size=128, batch_size=64):

    if not os.path.exists(test_root):
        raise FileNotFoundError(f"Test folder not found: {test_root}")
    
    _, eval_tf = get_transforms(img_size, augment=False)
    test_ds = datasets.ImageFolder(test_root, transform=eval_tf)
   
    print(f"Test classes:  {test_ds.class_to_idx}")
    print(f"Train classes: {class_names}")
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, 
                              shuffle=False, num_workers=0)
    
    model.eval()
    trues, preds = predict_loader(model, test_loader, device)
    metrics = summarize(trues, preds, class_names)
    
    print(f"\n Test evaluation on {len(test_ds)} images")
    print(f"  macro_F1: {metrics['macro_f1']:.4f}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    
    return metrics

def rank_confused_pairs(cm_norm, class_names, top=5):
    n = len(class_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((class_names[i], class_names[j], cm_norm[i, j] + cm_norm[j, i]))
    return sorted(pairs, key=lambda x: x[2], reverse=True)[:top]

@torch.no_grad()
def evaluate_unseen(model, test_root, device, class_names,
                    img_size=128, batch_size=64):
    _, eval_tf = get_transforms(img_size, augment=False)
    exts = ('.jpg', '.jpeg', '.png')
    paths = [p for p in Path(test_root).iterdir() if p.suffix.lower() in exts]
    
    if not paths:
        raise FileNotFoundError(f"No images found in {test_root}")
    
    print(f"Found {len(paths)} images in {test_root}")
    
    all_probs = []
    for p in paths:
        with Image.open(p) as img:
            tensor = eval_tf(img.convert('RGB')).unsqueeze(0).to(device)
        probs = torch.softmax(model(tensor), dim=1)[0]
        all_probs.append(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    max_confs = all_probs.max(axis=1)
    preds = all_probs.argmax(axis=1)
    
    print(f"\n═══ Unseen Class Analysis ({len(paths)} images) ═══")
    print(f"Mean confidence:   {max_confs.mean():.3f}")
    print(f"Median confidence: {np.median(max_confs):.3f}")
    print(f"% conf < 0.5:      {(max_confs < 0.5).mean()*100:.1f}%")
    print(f"% conf < 0.7:      {(max_confs < 0.7).mean()*100:.1f}%")
    
    pred_names = [class_names[i] for i in preds]
    unique, counts = np.unique(pred_names, return_counts=True)
    print(f"\nPredicted classes (should be spread if model is uncertain):")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {u}: {c}")
    
    return {
        'n': len(paths),
        'mean_conf': float(max_confs.mean()),
        'median_conf': float(np.median(max_confs)),
        'frac_below_0.5': float((max_confs < 0.5).mean()),
        'frac_below_0.7': float((max_confs < 0.7).mean()),
        'probs': all_probs,
        'preds': preds,
        'pred_names': pred_names,
    }