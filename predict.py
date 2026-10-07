import json
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.model import TLResNet   # ← این خط اضافه شد


DEFAULT_CONFIG = {
    'architecture': 'resnet18',
    'num_classes': 8,
    'img_size': 224,
    'threshold': 0.70,
    'checkpoint': '../checkpoints/resnet_ft.pt',
    'imagenet_mean': [0.485, 0.456, 0.406],
    'imagenet_std': [0.229, 0.224, 0.225],
    'class_names': [
        'ambulance', 'autobus', 'kamyun', 'kamyunet',
        'minibus', 'savari', 'taxi', 'vanet',
    ],
}


def build_model(architecture='resnet18', num_classes=8):
    if architecture == 'resnet18':
        return TLResNet(num_classes=num_classes, trainable=('fc', 'layer4'))
    raise ValueError(f"Unknown architecture: {architecture}")


def build_transform(img_size=224, mean=None, std=None):
    mean = mean or DEFAULT_CONFIG['imagenet_mean']
    std = std or DEFAULT_CONFIG['imagenet_std']
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    meta = {
        'architecture': ckpt.get('architecture', DEFAULT_CONFIG['architecture']),
        'num_classes': DEFAULT_CONFIG['num_classes'],
        'img_size': ckpt.get('img_size', DEFAULT_CONFIG['img_size']),
        'threshold': ckpt.get('threshold', DEFAULT_CONFIG['threshold']),
        'class_names': ckpt.get('class_names', DEFAULT_CONFIG['class_names']),
        'seed': ckpt.get('seed', None),
        'config': ckpt.get('config', {}),
    }

    model = build_model(meta['architecture'], meta['num_classes'])
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model, meta


_model_cache = None
_meta_cache = None


def _get_model(checkpoint=None):
    global _model_cache, _meta_cache
    if _model_cache is None:
        path = checkpoint or DEFAULT_CONFIG['checkpoint']
        _model_cache, _meta_cache = load_checkpoint(path)
    return _model_cache, _meta_cache


@torch.no_grad()
def predict(image_path, checkpoint=None, threshold=None):
    model, meta = _get_model(checkpoint)

    if threshold is None:
        threshold = meta['threshold']

    class_names = meta['class_names']
    img_size = meta['img_size']

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = build_transform(img_size)
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')
        tensor = transform(img_rgb).unsqueeze(0)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    predicted_class = class_names[pred_idx]

    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence, 4),
        'probabilities': {c: round(float(p), 4) for c, p in zip(class_names, probs)},
        'needs_review': bool(confidence < threshold),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path> [checkpoint]")
        sys.exit(1)

    image_path = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else None

    result = predict(image_path, checkpoint=checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()