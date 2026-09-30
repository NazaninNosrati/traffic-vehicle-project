import hashlib
import os

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


def md5_hash(filepath):
    with open(filepath , 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def average_hash(filepath , hash_size=8):

    try:
        with Image.open(filepath) as img:
            img= img.convert('L').resize((hash_size , hash_size) , Image.LANCZOS)
            pixels = np.asarray(img , dtype=np.float64)
        avg= pixels.mean()
        bits=(pixels > avg).flatten()
        return ''.join('1' if b else '0' for b in bits)
    except Exception:
        return None

def scan_all_images(data_root , splits=('train' , 'test' , 'unclean')):

    all_files= []
    for split in splits:
        split_path=os.path.join(data_root , split)
        for class_name in os.listdir(split_path):
            class_dir=os.path.join(split_path , class_name)    
            if not os.path.isdir (class_dir):
                continue
            for fname in os.listdir(class_dir):
                all_files.append((split , class_name , fname , os.path.join(class_dir , fname)))

    records= []
    for split , class_name ,fname , fpath , in tqdm(all_files , desc="hashing images"):
        records.append({
            'split' : split ,
            'class' : class_name ,
            'filename' : fname ,
            'path' : fpath ,
            'md5' : md5_hash(fpath),
            'avghash' : average_hash(fpath),
        })
    return records




feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor.fc= torch.nn.Identity()
feature_extractor.eval()

embed_transform = transforms.Compose([
    transforms.Resize((224 , 224)) ,
    transforms.ToTensor() ,
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std= [ 0.229 , 0.224 , 0.225]),
])

_extractor = None

def _get_extractor():
    global _extractor
    if _extractor is None:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        m.eval()
        _extractor = m
    return _extractor

@torch.no_grad()
def get_embedding(filepath):
    with Image.open(filepath) as img:
        tensor = embed_transform(img.convert('RGB')).unsqueeze(0)
    return _get_extractor()(tensor).squeeze(0).numpy()
