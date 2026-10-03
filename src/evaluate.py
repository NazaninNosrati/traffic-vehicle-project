import torch
from sklearn.metrics import precision_recall_fscore_support


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
    p, r, f1, _ = precision_recall_fscore_support(
        trues, preds, labels=list(range(len(class_names))),
        average=None, zero_division=0)
    row = {'macro_precision': p.mean(), 'macro_recall': r.mean(),
           'macro_f1': f1.mean(),
           'lowest_recall_class': class_names[int(r.argmin())],
           'lowest_precision_class': class_names[int(p.argmin())]}
    for i, c in enumerate(class_names):
        row[f'recall_{c}'] = r[i]
        row[f'precision_{c}'] = p[i]
    return row
