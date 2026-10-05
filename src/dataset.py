import hashlib
import os
import shutil

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
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

def scan_all_images(data_root , splits=(None)):

    if splits is None:
        splits = ('train', 'unclean')
        test_path = os.path.join(data_root, 'test')
        if os.path.exists(test_path):
            splits = ('train', 'test', 'unclean')

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



def make_train_val_split(df, val_size=0.2, seed=42):
    train_all = df[df['split'] == 'train'].reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(sss.split(train_all, train_all['class']))
    return (train_all.iloc[tr_idx].reset_index(drop=True),
            train_all.iloc[va_idx].reset_index(drop=True))


def export_split(part_df, out_root):
    for _, row in part_df.iterrows():   
        dest_dir = os.path.join(out_root, row['class'])
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(row['path'], os.path.join(dest_dir, row['filename']))


def build_processed(train_part, val_part, out_root):
    for name in ('train', 'val'):
        shutil.rmtree(os.path.join(out_root, name), ignore_errors=True)
    export_split(train_part, os.path.join(out_root, 'train'))
    export_split(val_part, os.path.join(out_root, 'val'))


def md5_overlap(a_df, b_df):
    return len(set(a_df['md5']) & set(b_df['md5']))


def check_no_leakage(train_part, val_part, test_part=None):
    result = {
        'train∩val': md5_overlap(train_part, val_part),
    }
    if test_part is not None:
        result['train∩test'] = md5_overlap(train_part, test_part)
        result['val∩test'] = md5_overlap(val_part, test_part)
    return result


def make_imagefolders(processed_root , test_root=None ):
    tr = datasets.ImageFolder(os.path.join(processed_root, 'train'))
    va = datasets.ImageFolder(os.path.join(processed_root, 'val'))

    if test_root is None or not os.path.exists(test_root):
        return tr, va, None
    te = datasets.ImageFolder(test_root)
    assert tr.class_to_idx == va.class_to_idx== te.class_to_idx , "class_to_idx mismatch"
    return tr, va, te 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size, augment=False):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(), norm])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(), norm])
    return train_tf, eval_tf

def make_loaders(root, img_size, batch_size, seed, augment=False):
    train_tf, eval_tf = get_transforms(img_size, augment)
    train_ds = datasets.ImageFolder(f'{root}/train', transform=train_tf)
    val_ds = datasets.ImageFolder(f'{root}/val', transform=eval_tf)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    return train_ds, val_ds, train_loader, val_loader

def simulate_imbalance(targets, keep_fraction, seed):
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    keep = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        n = max(1, round(len(idx) * keep_fraction.get(int(c), 1.0)))
        keep.extend(rng.choice(idx, size=n, replace=False).tolist())
    return sorted(keep)