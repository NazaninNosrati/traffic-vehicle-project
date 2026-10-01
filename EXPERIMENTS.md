# Experiment Log

## — Data audit: duplicate & label-conflict detection

Changed: n/a (data cleaning stage, not a training run)
Fixed: n/a
Method: MD5 (exact-copy) + average hash (hash_size=16, perceptual) across train/test/unclean (4360 images total)
Result: MD5: 24 duplicate groups, 1 label conflict. Average hash: 46 candidate groups, 8 label-conflict candidates.
Findings:
- MD5 conflict: same Nissan pickup image labeled vanet in train and neysan in unclean — evidence that vanet in train was mislabeling actual neysan vehicles. Removed from train/vanet, kept as neysan in unclean.
- Average hash: most of the 8 candidate groups were false positives (similar camera angle/background but different vehicle — e.g. two taxis with different livery, or taxi vs. sedan). Perceptual hash matches overall brightness pattern, not vehicle identity — none of these removed without manual confirmation.
Notes: hash_size=16 used (not 8) to reduce false-positive rate on near-identical camera frames.

##  — Data audit: mislabel suspects via ResNet18 embeddings

Changed: n/a (data cleaning stage, not a training run)
Fixed: n/a
Method: ImageNet-pretrained ResNet18 (no training on our data, fc replaced by Identity) used as a fixed feature extractor. For each train image, a 512-d embedding was computed; per class, the centroid was calculated and each image's Euclidean distance to its own class centroid was recorded. The 15 farthest images per class were reviewed by eye.
Why: Hashing only finds duplicates; it cannot find a unique image that was placed in the wrong class.
Result: <1928> train images embedded, <5> unreadable/skipped. Reviewed 15 suspects x 8 classes. Confirmed problems: <3> wrong-class images moved, <3> images excluded (belong to no class: e.g. van, police car, hearse), <15> left in place (unusual but valid: dark, odd angle).
Decisions:
- Images matching no class were moved to data/excluded/ (not deleted) and logged in results/manual_relabel_log.csv with a reason.
- Dark or low-quality taxis were kept: color is not the class definition, and such images occur in real deployment.
- Hard pairs (minibus/autobus, kamyun/kamyunet) were NOT bulk-relabeled by eye; only clearly wrong cases were moved. These pairs are left for the confusion-matrix / pair_confusion analysis.
Limitations: Distance to centroid flags visually unusual images, not necessarily wrong labels. Classes with high internal variety (e.g. savari) produce more false suspects. The final decision on every image was manual.
Notes: <_>

## — Train/validation split

**Changed:** n/a (data preparation)
**Method:** StratifiedShuffleSplit on cleaned train, test_size=0.2, seed=42. Split exported as image folders (data/processed/train, data/processed/val, copied from raw) plus file lists (train_files.csv, val_files.csv). Test folder frozen; its file list + MD5 saved to test_files.csv.
**Result:** train=<1539>, val=<385>, test=<400>.
**Decisions:** unclean not used in training (analysis only, day 5). No Nissan/neysan images in train or val.
**Limitations:** val has only ~<50> images per class, so small metric differences between experiments are within noise.
**Notes:** 0 = ambulance --> 44 , 4 = minibus --> 45

## — CNN baseline: setup

**Changed:** n/a (baseline definition)
**Model:** CNNBaseline, 3 blocks (Conv3x3 -> ReLU -> MaxPool2), channels 32/64/128, global average pooling, Dropout(0), Linear(128, 8); 94,280 parameters.
**Data/transforms:** 128x128 input, ImageNet mean/std normalization, no augmentation (train and val use the same transform). Train from data/processed/train, val from data/processed/val. Test not loaded.
**Seed:** 42 (random, numpy, torch; DataLoader shuffle uses a seeded generator).
**Notes:** BatchNorm left out on purpose to keep the baseline minimal; num_workers=0 (Windows).