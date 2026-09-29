import hashlib
import os

import numpy as np
from PIL import Image
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



