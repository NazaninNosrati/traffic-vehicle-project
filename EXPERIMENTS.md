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

## Exp 01 — CNN baseline: results

**Changed:** n/a (baseline run)
**Budget:** <40> epochs, Adam lr=1e-3 (fixed), batch 32, no scheduler.
**Best checkpoint:** epoch <40>, val macro-F1 = <0.69>, val loss at best = <0.91>.
**Final epoch:** train loss < 0.6>, val loss <0.91>, val macro-F1 <0.69>.

## Exp 02 — CNN baseline: results

**Changed:** n/a (baseline run)
**Budget:** <80> epochs, Adam lr=1e-3 (fixed), batch 32, no scheduler.
**Best checkpoint:** epoch <40>, val macro-F1 = <0.75>, val loss at best = <0.76>.
**Final epoch:** train loss <0.31>, val loss <0.75>, val macro-F1 <0.76>.

## — Baseline

**Config:** run `baseline` (results/experiments.csv)
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 30 epochs, batch 32, CrossEntropy, no augmentation, no dropout, max-pooling
**Result:** macro_P=0.671, macro_R=0.660, macro_F1=0.656
**Best epoch / gap:** 28 / val_loss − train_loss = 0.989 − 0.896 = 0.094
**Lowest-recall class:** minibus (0.444)  **Lowest-precision class:** kamyunet (0.468)
**Notes:** Baseline converged at epoch 28. Best F1 = 0.656. Gap is small (0.094), suggesting no significant overfitting. Weakest classes: minibus (recall) and kamyunet (both P and R ~0.468), indicating confusion between similar vehicle types (minibus/vanet, kamyun/kamyunet).؟

## — Ablation: augmentation

**Changed:** augmentation off → on (RandomResizedCrop, HorizontalFlip, Rotation 10°, ColorJitter)
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 30 epochs, batch 32, CrossEntropy
**Config:** run `aug` (results/experiments.csv)
**Result:** macro_P=0.653, macro_R=0.614, macro_F1=0.620 (baseline: P=0.671, R=0.660, F1=0.656)
**Best epoch / gap:** 28 / val_loss − train_loss = 1.105 − 1.171 = −0.066 (baseline: 28 / 0.094)
**Lowest-recall class:** vanet (0.460)  **Lowest-precision class:** kamyunet (0.377)
**Notes:** Augmentation did NOT help — macro_F1 dropped by 0.036 (from 0.656 to 0.620). The drop exceeds the noise threshold (~0.03). Train loss increased (0.896 → 1.171), which is expected with augmentation, but val_loss also increased (0.989 → 1.105), so the model did not generalize better. Notably, the gap flipped sign (−0.066), meaning val_loss is now lower than train_loss — this is normal with heavy augmentation but does NOT indicate improvement here. Some classes improved (savari recall 0.80→0.82, minibus recall 0.44→0.60), while others degraded sharply (ambulance recall 0.80→0.59, autobus 0.69→0.57). The augmentation may be too aggressive for this dataset size, or the 30-epoch budget is insufficient for the model to benefit.

## — Ablation: dropout (p=0.3)

**Changed:** dropout_p 0.0 → 0.3 (in classifier head, before final Linear)
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 30 epochs, batch 32, CrossEntropy, no augmentation, max-pooling
**Config:** run `dropout_0.3` (results/experiments.csv)
**Result:** macro_P=0.631, macro_R=0.613, macro_F1=0.608 (baseline: P=0.671, R=0.660, F1=0.656)
**Best epoch / gap:** 28 / val_loss − train_loss = 1.093 − 1.077 = 0.016 (baseline: 28 / 0.094)
**Lowest-recall class:** kamyunet (0.362)  **Lowest-precision class:** kamyunet (0.415)
**Notes:** Dropout 0.3 did NOT help — macro_F1 dropped by 0.048 (from 0.656 to 0.608), well above the noise threshold (~0.03). The gap shrank dramatically (0.094 → 0.016), confirming dropout reduced overfitting, but this came at the cost of overall accuracy — the model became under-confident and lost both precision and recall. The model is likely UNDERFITTING, not overfitting: baseline gap was already small (0.094), so regularization hurt more than helped. Notable regression: autobus recall dropped 0.69 → 0.41 (−0.28), the largest single-class drop across all experiments. Slight improvements: minibus recall 0.44 → 0.71 (+0.27). Overall, dropout 0.3 is not recommended for this setup.

## — Ablation: dropout (p=0.5)

**Changed:** dropout_p 0.0 → 0.5 (in classifier head)
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 30 epochs, batch 32, CrossEntropy, no augmentation, max-pooling
**Config:** run `dropout_0.5` (results/experiments.csv)
**Result:** macro_P=0.638, macro_R=0.625, macro_F1=0.621 (baseline: P=0.671, R=0.660, F1=0.656)
**Best epoch / gap:** 28 / val_loss − train_loss = 1.106 − 1.131 = −0.025 (baseline: 28 / 0.094)
**Lowest-recall class:** kamyunet (0.362)  **Lowest-precision class:** kamyunet (0.405)
**Notes:** Dropout 0.5 also did NOT help — macro_F1 dropped by 0.035 (from 0.656 to 0.621), slightly above the noise threshold (~0.03). Interestingly, F1 (0.621) is marginally BETTER than dropout 0.3 (0.608), suggesting the model tolerates stronger dropout better than mild dropout here, but both remain below baseline. The gap flipped negative (−0.025), indicating val_loss is now lower than train_loss — a sign of heavy regularization. Per-class analysis shows drop_0.5 recovers some classes that drop_0.3 hurt (vanet recall 0.48 → 0.58, autobus slightly up 0.41 → 0.43), while ambulance remains degraded (0.68 → 0.59 vs baseline 0.80). Overall, dropout in the classifier head does not improve this model — the baseline is already well-calibrated.

## — Ablation: pooling (max → avg)

**Changed:** pool_type max → avg (MaxPool2d → AvgPool2d in all 3 conv blocks)
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 30 epochs, batch 32, CrossEntropy, no augmentation, no dropout
**Config:** run `pool_avg` (results/experiments.csv)
**Result:** macro_P=0.612, macro_R=0.609, macro_F1=0.597 (baseline: P=0.671, R=0.660, F1=0.656)
**Best epoch / gap:** 30 / val_loss − train_loss = 1.110 − 1.044 = 0.067 (baseline: 28 / 0.094)
**Lowest-recall class:** kamyunet (0.234)  **Lowest-precision class:** autobus (0.482)
**Notes:** Avg-pooling performed WORST among all ablations so far — macro_F1 dropped by 0.059 (from 0.656 to 0.597), nearly double the noise threshold (~0.03). The model was still improving at epoch 30 (best epoch = 30, the last epoch), suggesting it was UNDERTRAINED — avg-pooling likely slowed convergence. kamyunet collapsed badly (recall 0.468 → 0.234), the worst single-class performance across all experiments. taxi remained strongest (recall 0.92). Interpretation: max-pooling preserves the strongest activations (edges, corners, distinctive features), which matters for distinguishing visually similar vehicle types. Avg-pooling smooths these out, hurting fine-grained discrimination.

## — Ablation: weight decay (1e-4)

**Changed:** weight_decay 0.0 → 1e-4 (AdamW, applied to all parameters)
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 30 epochs, batch 32, CrossEntropy, no augmentation, no dropout, max-pooling
**Config:** run `wd_1e-4` (results/experiments.csv)
**Result:** macro_P=0.652, macro_R=0.634, macro_F1=0.630 (baseline: P=0.671, R=0.660, F1=0.656)
**Best epoch / gap:** 28 / val_loss − train_loss = 0.988 − 0.900 = 0.088 (baseline: 28 / 0.094)
**Lowest-recall class:** minibus (0.422)  **Lowest-precision class:** kamyunet (0.476)
**Notes:** Weight decay 1e-4 slightly reduced macro_F1 by 0.026 (from 0.656 to 0.630), which is just below the noise threshold (~0.03) — so it is INCONCLUSIVE whether the drop is real or noise. The gap barely moved (0.094 → 0.088), suggesting weight decay at this magnitude barely regularizes. Train and val losses are almost identical to baseline (0.900 vs 0.896, 0.988 vs 0.989). Per-class patterns are similar to baseline: taxi remains strongest (F1 ≈ 0.877), kamyunet remains weakest. Ambulance recall stays high (0.77) but precision drops (0.56 vs baseline 0.59). This ablation is essentially a NO-OP: 1e-4 is too small to matter for this model. Larger values (1e-3, 1e-2) might show a real effect, but given all prior ablations failed, the bottleneck is likely model CAPACITY, not regularization.

## — Ablation: scheduler (none → ReduceLROnPlateau)

**Changed:** scheduler none → ReduceLROnPlateau (mode='min', factor=0.5, patience=3), stepped on val_loss
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 30 epochs, batch 32, CrossEntropy, no augmentation, no dropout, max-pooling, no weight decay
**Config:** run `sched_plateau` (results/experiments.csv)
**Result:** macro_P=0.662, macro_R=0.658, macro_F1=0.649 (baseline: P=0.671, R=0.660, F1=0.656)
**Best epoch / gap:** 30 / val_loss − train_loss = 0.939 − 0.845 = 0.094 (baseline: 28 / 0.094)
**Lowest-recall class:** kamyunet (0.362)  **Lowest-precision class:** kamyunet (0.486)
**Notes:** Plateau scheduler produced the BEST non-baseline result so far — macro_F1 = 0.649, only 0.007 below baseline. This is WELL within the noise threshold (~0.03), so the difference from baseline is NOT statistically significant. Crucially, the best epoch was 30 (the last epoch), meaning the model was STILL IMPROVING when training stopped — with more epochs, plateau would likely surpass baseline. Train loss dropped to 0.845 (lowest of all experiments), val_loss to 0.939 (second lowest), and gap stayed at 0.094 — same as baseline, indicating no overfitting trade-off. Per-class: kamyunet remains the weakest (recall 0.36), but savari precision jumped to 0.77 (vs baseline 0.63), the best across all experiments. Strong candidate for longer training.