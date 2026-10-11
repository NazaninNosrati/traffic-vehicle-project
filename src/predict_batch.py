import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.70   

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    'ambulance', 'autobus', 'kamyun', 'kamyunet',
    'minibus', 'savari', 'taxi', 'vanet',
]


class ResNetWrapper(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.net(x)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    if isinstance(ckpt, dict):
        state = ckpt.get('model_state_dict', ckpt)
        classes = ckpt.get('class_names', CLASS_NAMES)
        img_size = ckpt.get('img_size', 224)
        threshold = ckpt.get('threshold', THRESHOLD)
    else:
        state = ckpt
        classes = CLASS_NAMES
        img_size = 224
        threshold = THRESHOLD
    
    model = ResNetWrapper(num_classes=len(classes))
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    
    print(f"Model loaded: {len(classes)} classes")
    print(f"   img_size:  {img_size}")
    print(f"   threshold: {threshold}")
    
    return model, list(classes), img_size, threshold



def collect(root):
   
    root = Path(root)
    if not root.exists():
        print(f" Folder not found: {root}")
        return []
    items = []
    
    
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs:
        for cdir in sorted(subdirs):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in EXTS:
                    items.append((f, cdir.name))
        if items:
            return items
   
    for f in sorted(root.iterdir()):
        if f.is_file() and f.suffix.lower() in EXTS:
            items.append((f, None))
    
    return items


@torch.no_grad()
def predict_batch(model, classes, items, img_size, threshold, bs=32):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    rows = []
    for i in range(0, len(items), bs):
        chunk = items[i:i+bs]
        x = torch.stack([tfm(Image.open(p).convert("RGB")) for p, _ in chunk]).to(DEVICE)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        
        for (p, true), prob in zip(chunk, probs):
            k = int(prob.argmax())
            conf = float(prob[k])
            rows.append({
                "file": p.name,
                "path": str(p),
                "true": true if true else "",
                "predicted_class": classes[k],
                "confidence": round(conf, 4),
                "needs_review": conf < threshold,
                "probabilities": {c: round(float(pr), 4) for c, pr in zip(classes, prob)},
            })
    
    return rows


def evaluate(rows):
    """فقط روی نمونه‌هایی که true label دارن."""
    with_true = [r for r in rows if r["true"]]
    if not with_true:
        return None
    
    correct = sum(1 for r in with_true if r["predicted_class"] == r["true"])
    acc = correct / len(with_true)
    
    print(f"\n   Evaluation ({len(with_true)} samples with labels):")
    print(f"   Accuracy:  {correct}/{len(with_true)} = {acc:.1%}")
    
    print(f"\n  Per-class:")
    for c in sorted(set(r["true"] for r in with_true)):
        cr = [r for r in with_true if r["true"] == c]
        cr_correct = sum(1 for r in cr if r["predicted_class"] == c)
        print(f"    {c:<15} {cr_correct}/{len(cr)}")
    
    return acc


def main(tests_root, ckpt_path, out_dir="results/test_runs"):
    model, classes, img_size, threshold = load_model(ckpt_path)
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    items = collect(tests_root)
    print(f"\n Found {len(items)} images")
    
    if not items:
        print(" No images found")
        return
    
    rows = predict_batch(model, classes, items, img_size, threshold)
    
    csv_path = out / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "path", "true", "predicted_class",
            "confidence", "needs_review"
        ])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    

    json_path = out / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    
    print(f"\n Saved:")
    print(f"   {csv_path}")
    print(f"   {json_path}")
    
   
    evaluate(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="folder with test images")
    ap.add_argument("--ckpt", required=True, help="path to checkpoint")
    ap.add_argument("--out", default="results/test_runs", help="output folder")
    a = ap.parse_args()
    
    main(a.tests, a.ckpt, a.out)