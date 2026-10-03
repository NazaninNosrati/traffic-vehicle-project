import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
