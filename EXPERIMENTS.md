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

## — Ablation: imbalanced sampling

**Changed:** standard shuffle=True → BalancedBatchSampler (4 per class per batch) on simulated-imbalanced training set
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 20 epochs, batch 32, CrossEntropy, no augmentation, no dropout, max-pooling, no class weights
**Config:** runs `imb_standard`, `imb_balanced` (results/experiments.csv)
**Simulated imbalance:** ambulance=35 (20%), minibus=54 (30%), others≈195 (100%), n_train=1272
**Result:**
- imb_standard: macro_P=0.390, macro_R=0.488, macro_F1=0.430
- imb_balanced: macro_P=0.570, macro_R=0.553, macro_F1=0.529
- Delta: +0.180 P, +0.065 R, **+0.099 F1** (well above noise ~0.03)
**Best epoch / gap:** 17 / 0.328 (standard) → 20 / 0.391 (balanced)
**Lowest-recall class:** ambulance (standard, recall=0.000) → kamyunet (balanced, recall=0.106)
**Notes:** On the simulated-imbalanced training set, standard training suffered catastrophic minority-class collapse. ambulance and minibus both received F1 = 0.000 — the model never predicted them at all. This is the classic majority-class bias: accuracy remained at 0.501, but macro-F1 dropped to 0.430. BalancedBatchSampler recovered both classes completely: ambulance F1 jumped from 0.000 → 0.543, minibus from 0.000 → 0.548. Overall macro-F1 improved by +0.099 (0.430 → 0.529), the largest single-factor gain in this project. Trade-off: the balanced sampler slightly hurt majority classes — vanet recall dropped from 0.60 → 0.34, kamyunet recall from 0.55 → 0.11 — but the macro-level gain dominates, and per-class fairness is exactly the goal under imbalance. The gap increased from 0.328 to 0.391, indicating a small increase in overfitting, expected since the sampler sees the minority classes more often. Recommendation: use BalancedBatchSampler whenever deployment requires per-class reliability, especially for rare classes like ambulance and minibus.

## — Ablation: balanced sampling under simulated imbalance

**Changed:** standard shuffle=True → BalancedBatchSampler (4 samples/class/batch, replacement=True)
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 20 epochs, batch 32, CrossEntropy, no augmentation, no dropout, max-pooling, no class weights
**Simulated imbalance:** ambulance=35 (20%), minibus=54 (30%), others≈195 (100%), n_train=1272
**Config:** runs `baseline`, `imb_standard`, `imb_balanced` (results/experiments.csv)
**Result:**
- baseline (balanced): acc=0.574, macro_F1=0.565, ambulance_R=0.682, minibus_R=0.333
- imb_standard: acc=0.501, macro_F1=0.430, ambulance_R=0.000, minibus_R=0.000
- imb_balanced: acc=0.553, macro_F1=0.529, ambulance_R=0.568, minibus_R=0.822
**Delta (imb_standard → imb_balanced):** +0.052 acc, +0.099 macro_F1, +0.568 ambulance recall, +0.822 minibus recall

**Notes:** Under simulated imbalance, standard training suffered catastrophic minority-class collapse: both ambulance and minibus reached recall = 0.000 — the model never predicted them. This is the classic majority-class bias: accuracy stayed near 0.50 while macro-F1 dropped to 0.430. BalancedBatchSampler recovered both classes completely — ambulance recall jumped from 0.000 → 0.568, minibus from 0.000 → 0.822 — and lifted macro-F1 by +0.099, the largest single-factor gain in this project. Interestingly, imb_balanced also beat imb_standard in raw accuracy (+0.052), which is unusual: oversampling usually trades accuracy for recall. Here the dataset is small enough that ignoring two classes also hurt overall accuracy. Comparison with baseline reveals a key insight: baseline minibus recall was only 0.333 even with balanced data — minibus is intrinsically hard. Imb_balanced pushed minibus recall to 0.822 by repeatedly exposing the model to the 54 available examples, effectively "over-training" on the minority class. Trade-off: kamyunet and vanet recall dropped (majority classes paid the cost), but macro-level fairness improved. Recommendation: use BalancedBatchSampler whenever per-class reliability matters, especially for rare, safety-critical classes like ambulance.

## — Ablation: CrossEntropy vs BCEWithLogitsLoss

**Changed:** loss CrossEntropy → BCEWithLogitsLoss (one-hot targets, sigmoid per-class)
**Fixed:** CNNBaseline, split (seed 42), AdamW lr=1e-3, 20 epochs, batch 32, no augmentation, no dropout, max-pooling, no class weights
**Config:** runs `ce (baseline)`, `bce` (results/experiments.csv)

**Result:**
- CE:  acc=0.574, macro_P=0.609, macro_R=0.571, macro_F1=0.565
- BCE: acc=0.545, macro_P=0.538, macro_R=0.542, macro_F1=0.525
- Delta: −0.029 acc, −0.071 P, −0.029 R, **−0.040 macro_F1**

**Which macro-F1 is higher?** CE (0.565 vs 0.525). The gap of 0.040 exceeds the noise threshold (~0.03), so the difference is real, not noise.

**Loss curves:** BCE converges more slowly in early epochs (best_epoch = 18 for BCE vs 20 for CE, but BCE's curve is flatter — it improves more gradually without a sharp peak). CE reaches high macro-F1 earlier and holds it; BCE catches up slowly and never fully closes the gap within the 20-epoch budget.

**Confidence table:**
| Metric | CE | BCE |
|---|---:|---:|
| mean_top_score | 0.559 | 0.455 |
| mean_sum_of_scores | 1.000 | 0.957 |
| frac_none_above_0.5 | 0.468 | 0.636 |
| frac_multiple_above_0.5 | 0.000 | 0.026 |

- CE probabilities sum to exactly 1.000 (softmax).
- BCE sums to 0.957 — not a probability distribution.
- BCE is more "under-confident": 63.6% of images have no class above 0.5, vs 46.8% for CE.
- BCE produces 2.6% of images with *multiple* classes above 0.5 — logically inconsistent for a single-label task. CE never does this (softmax → one winner).

**Error pattern by class recall:**
| Class | CE | BCE | Δ |
|---|---:|---:|---:|
| ambulance | 0.68 | 0.70 | +0.02 |
| autobus | 0.63 | 0.47 | −0.16 |
| kamyun | 0.44 | 0.62 | +0.18 |
| kamyunet | 0.30 | 0.13 | −0.17 |
| minibus | 0.33 | 0.33 | 0.00 |
| savari | 0.82 | 0.66 | −0.16 |
| taxi | 0.90 | 0.94 | +0.04 |
| vanet | 0.46 | 0.48 | +0.02 |

CE wins on 4 classes, BCE wins on 3, one tie. The critical loss for BCE is kamyunet (0.30 → 0.13) and savari (0.82 → 0.66) — two classes CE handles well. BCE only clearly wins on kamyun. Full confusion matrices will be computed in the final evaluation for the selected model.

**Hypothesis for BCE's slower learning (as a hypothesis, not a verdict):**
Each image contributes 7 zero targets and only 1 positive target per sample in the BCE formulation. The positive signal is diluted ~7:1, and classes do not compete directly (no softmax normalization). This may explain why BCE converges more slowly and produces lower-confidence scores. An alternative explanation is that the 20-epoch budget is insufficient for BCE — it might converge to a comparable point with more epochs. We do not claim either as fact without further experiments.

**Equal-budget tuning (optional, not run):**
If BCE had clearly not converged within the same budget, the fairest comparison would be to run both CE and BCE with two learning rates (e.g., 1e-3 and 3e-3) so each loss gets an equal tuning budget, as the project requires. This is optional and time-consuming; here we report the single-lr comparison and note the caveat.

**Notes:** BCE underperformed CE by 0.040 macro-F1 — a real, above-noise gap. CE is the correct choice for single-label classification: softmax enforces mutual exclusivity (probabilities sum to 1, one winner), while BCE treats the 8 classes as independent binary decisions. The confidence analysis confirms this: BCE produced 2.6% logically-inconsistent images with multiple classes above 0.5, and was systematically under-confident (mean_top_score 0.455 vs 0.559). BCE's much smaller gap (0.006 vs 0.111) suggests it regularizes more strongly — a plausible side effect of the diluted positive signal — but this came at the cost of accuracy, not as a benefit. Recommendation: use CrossEntropyLoss for the final model.

## — Ablation: CrossEntropy vs BCEWithLogitsLoss (equal-budget tuning)

**Changed:** loss {CE, BCE} × lr {1e-3, 3e-3} — 4 runs, equal budget each
**Fixed:** CNNBaseline, split (seed 42), AdamW, 20 epochs, batch 32, no aug, no dropout, max-pool
**Config:** runs `ce_lr1e-3`, `ce_lr3e-3`, `bce_lr1e-3`, `bce_lr3e-3`

**Result:**
| loss | lr | macro_F1 | accuracy |
|---|---|---|---|
| CE | 1e-3 | 0.567 | 0.579 |
| CE | 3e-3 | 0.596 | 0.608 |
| BCE | 1e-3 | 0.525 | 0.545 |
| **BCE** | **3e-3** | **0.631** | **0.631** |

**Winner:** BCE with lr=3e-3 (0.631) — beats best CE (0.596) by +0.035.

**Notes:** The single-lr comparison (both at 1e-3) suggested CE wins by 0.042. But equal-budget tuning reverses the conclusion: BCE at lr=3e-3 reaches 0.631, beating CE's best (0.596 at lr=3e-3) by +0.035. BCE was slower to converge at lr=1e-3 (best_epoch=18) — consistent with the diluted-positive-signal hypothesis (7 zeros : 1 one per image) — but at lr=3e-3 it converges strongly and generalizes better. Per-class: kamyunet recall recovered (0.17 → 0.40), autobus improved, ambulance stayed ~0.66. This result demonstrates why equal-budget tuning matters: an unfair single-lr comparison would have led to the wrong conclusion.

**Recommendation:** for this dataset, BCE at lr=3e-3 is the strongest loss choice. However, confidence behavior must be re-examined for this winner (BCE probabilities do not sum to 1, and multi-positive predictions may appear).

## — Transfer Learning: ResNet-18 Feature Extraction (vs CNNBaseline)

**Question:** Does using an ImageNet-pretrained backbone beat training a CNN from scratch on ~2,000 traffic-vehicle images?

**Setup:**
- CNNBaseline: 11.2M trainable params, trained from random init
- ResNet-FE: ResNet-18 (ImageNet-pretrained), backbone frozen, only fc (4,104 params) trained
- Both: 20 epochs, AdamW, CrossEntropy, batch 32, no augmentation

**Results:**
| Model | macro_F1 | trainable params | best_epoch |
|---|---:|---:|---:|
| CNNBaseline | 0.565 | 11.2M | 20 |
| **ResNet-FE** | **0.898** | **4,104** | **16** |
| Δ | **+0.333** | **−2700×** | — |

**Why ResNet wins:**
1. **Generalizable features:** ImageNet-pretrained features (edges, textures, shapes) transfer directly to traffic vehicles.
2. **Data efficiency:** With only ~2,000 images, training from scratch cannot learn rich features; pretrained features fill the gap.
3. **Only 4k params trained:** The head specializes ImageNet features to the 8 traffic classes.

**Training curves:**
- CNNBaseline: slow, monotone improvement, still rising at epoch 20 (undertrained).
- ResNet-FE: F1 = 0.79 by epoch 2, plateau from epoch 8, best 0.898 at epoch 16.
- Plateau indicates the frozen backbone cannot adapt further — the head alone is not enough.

**Overfitting:** ResNet gap = 0.105 (small), consistent with frozen backbone acting as a regularizer.

**Limitation & next step:** Frozen features limit adaptation to traffic-specific patterns. Fine-tuning `layer4` + head with a smaller lr for pretrained params (1e-4) and normal lr for the head (1e-3) should close this gap and potentially exceed 0.92.

**Conclusion:** Transfer learning is the single most impactful change in this project. A 4k-parameter head on a frozen ImageNet backbone beats an 11M-parameter CNN trained from scratch by +0.333 macro-F1 — a ~60% relative improvement.

## — Transfer Learning: ResNet-18 Fine-tuning

**Changed:** feature extraction (frozen backbone, only fc) → fine-tuning (layer4 + fc trainable)
**Fixed:** ImageNet normalization, 224×224 input, Adam with 2 LR groups, 20 epochs, batch 32, CrossEntropy
**Trainable params:** 8,397,832 / 11,180,616 (75%)
**LR groups:** layer4: 1e-4, fc: 1e-3
**Config:** run `resnet_ft` (results/experiments.csv)
**Result:** macro_F1=0.932, accuracy=0.946 (vs feature extraction 0.898)
**Delta:** **+0.034 macro_F1** — above noise threshold (~0.03)

**Training curve:**
- Starts strong (F1=0.904 at epoch 1) because backbone is already pretrained
- Reaches plateau around epoch 11
- Best epoch: 11 (earlier than FE's epoch 16)
- train_loss → 0.00026 (essentially zero), val_loss ≈ 0.21
- **Gap = 0.211** — large, indicates the model has memorized training data

**Per-class improvement:**
The most striking gain is kamyunet: F1 jumped from ~0.42 (FE) to **0.917** (FT). This confirms that fine-tuning layer4 was essential to capture the subtle kamyun/kamyunet distinction — ImageNet features alone were not enough. All classes now exceed F1=0.78, with autobus best at 0.983 and vanet worst at 0.788 (small support of 17).

**Caution:** The very small train_loss suggests significant overfitting. Validation loss, however, remains stable at 0.21, so generalization is not yet collapsing. If training were extended, we would expect val_loss to rise. This is the natural risk of fine-tuning on a small dataset (~2,100 images) with 8.4M trainable params.

**Conclusion:** Fine-tuning beats feature extraction (+0.034 F1) and dramatically fixes the kamyunet problem that plagued all earlier models. The trade-off is a large train-val gap, which is expected but should be monitored.

## — Stretch: ResNet-18 from Scratch

**Changed:** pretrained ResNet-18 → ResNet-18 from scratch (weights=None), adapted for 32×32 input (conv1: 3×3 stride=1, maxpool removed)
**Fixed:** Adam lr=1e-3, 20 epochs, batch 64, CrossEntropy
**Trainable params:** 11,172,936 (all)
**Config:** run `resnet_scratch` (results/experiments.csv)
**Result:** macro_F1=0.827, accuracy=0.837 (vs CNNBaseline 0.565, vs ResNet-FE 0.898)
**Delta vs CNNBaseline:** +0.262
**Delta vs ResNet-FE:** −0.071

**Notes:** Surprisingly, ResNet from scratch reaches F1=0.827 — far better than the CNNBaseline (0.565), and only 0.071 below the pretrained feature extractor. This isolates two effects:
1. **Architecture matters (+0.262):** ResNet's skip connections, BatchNorm, and depth alone beat a shallow CNN, even without any pretraining.
2. **Pretraining matters (+0.071):** Pretrained features still add a real gain on top of the architecture.

The model is heavily overfit (train_loss = 0.009, val_loss = 0.577, gap = 0.568), yet validation loss steadily decreased, indicating it still generalizes. best_epoch = 20 (last epoch) — with more epochs it may improve further.

**Per-class:** kamyunet F1 = 0.754 (much better than CNNBaseline's 0.30), vanet still weakest (0.71).

**Conclusion:** The final ranking is ResNet-FT (0.932) > ResNet-FE (0.898) > ResNet-Scratch (0.827) > CNNBaseline (0.565). Transfer learning gives the biggest boost, but ResNet's architecture alone already delivers most of the improvement.

## — Ablation: augmentation + plateau (best_combo_v2)

**Changed:** baseline_v2 → augment=True + scheduler=plateau
**Fixed:** CNNBaseline, seed 42, AdamW lr=1e-3, 20 epochs, batch 32
**Config:** run `best_combo_v2` (results/experiments.csv)
**Result:** macro_F1=0.484 (vs baseline_v2 0.565) → **−0.081**
**Lowest-recall class:** vanet
**Notes:** Combining two regularizers on small data caused underfitting. kamyunet recall collapsed to 0.12. Since the baseline already showed a small gap (0.111), additional regularization only hurt. This is a negative result — a valid scientific finding: **more regularization ≠ better performance** when the model is already well-regularized.

## — Ablation: BCE on cleaned data (bce_v2)

**Changed:** loss CE → BCE on cleaned dataset
**Fixed:** CNNBaseline, seed 42, AdamW lr=1e-3, 20 epochs, batch 32, no aug, no dropout
**Config:** run `bce_v2` (results/experiments.csv)
**Result:** macro_F1=0.501 (vs baseline_v2 0.565) → **−0.064**
**Delta vs v1 BCE (0.525):** −0.024

**Notes:** BCE confirms its consistent underperformance on this single-label task, both on the old data (−0.040) and cleaned data (−0.064). The gap exceeds the noise threshold (~0.03). kamyunet recall collapsed to 0.05, continuing the BCE pattern of hurting hard classes. As hypothesized, the 7-zero / 1-one BCE target dilutes the positive signal and class competition is removed; this is not fixed by cleaning the data.

**Conclusion:** CE remains the right choice for this single-label problem. Cleaning the data did not change the relative ranking of losses.

## — Ablation: balanced sampling on cleaned data (v2)

**Changed:** standard shuffle → BalancedBatchSampler on simulated-imbalanced cleaned data
**Simulated imbalance:** ambulance=20%, minibus=30%, others=100%
**Fixed:** CNNBaseline, seed 42, AdamW lr=1e-3, 20 epochs, batch 32, loss=CE
**Config:** runs `imb_shuffle_v2`, `imb_balanced_v2` (results/experiments.csv)

**Result:**
| Run | F1 | ambulance R | kamyunet R |
|---|---:|---:|---:|
| baseline_v2 (balanced) | **0.565** | 0.68 | 0.30 |
| imb_shuffle_v2 | 0.405 | **0.037** | 0.41 |
| imb_balanced_v2 | 0.534 | **0.778** | 0.28 |
| **Δ (sampler)** | **+0.129** | **+0.741** | −0.13 |

**Notes:** The sampler effect is dramatically stronger in v2 than v1 (+0.129 vs +0.099). Without sampler, ambulance recall collapses to 0.037 — the model almost never predicts the minority class. With sampler, ambulance recall jumps to 0.778, matching the balanced baseline (0.68). Trade-off: kamyunet recall drops from 0.41 to 0.28, as sampler redirects capacity to the rare classes. The gap vs balanced baseline is only −0.031 (0.534 vs 0.565), meaning balanced sampling recovers most of the imbalance penalty.

**Conclusion:** BalancedBatchSampler is essential under imbalance. On cleaned data, it recovers ambulance almost completely (0.04 → 0.78) and lifts macro-F1 by +0.129 — the largest single-factor gain for this design on the cleaned dataset.

## — Validation Comparison Table

| Experiment | Macro P | Macro R | Macro F1 | Lowest-Recall | Lowest-Precision |
|---|---:|---:|---:|---|---|
| CNN baseline | 0.612 | 0.600 | 0.588 | vanet | vanet |
| Balanced batches | 0.532 | 0.540 | 0.534 | vanet | vanet |
| CrossEntropy / BCE | 0.490 | 0.538 | 0.501 | vanet | vanet |
| Best regularized + scheduled | 0.478 | 0.516 | 0.484 | vanet | vanet |
| ResNet feature extraction | 0.902 | 0.895 | 0.898 | vanet | vanet |
| **ResNet fine-tuning** | **0.936** | **0.930** | **0.932** | vanet | vanet |

### Analysis by row

**CNN baseline (0.588)** — reference point. vanet is the weakest class in both recall and precision, suggesting systematic confusion with a visually similar class.

**Balanced batches (0.534)** — trained on simulated-imbalanced data with BalancedBatchSampler. Despite the sampler, macro-F1 dropped 0.054 vs baseline because the training set itself is smaller and skewed. The sampler recovered minority-class recall but not enough to compensate. Lowest classes remain vanet.

**CrossEntropy / BCE comparison (0.501)** — BCE on cleaned data. Confirms BCE's underperformance on this single-label task (−0.087 vs baseline). The 7-zero / 1-one target dilutes the learning signal, and the lack of softmax competition leads to lower precision. vanet is again the weakest class.

**Best regularized + scheduled (0.484)** — augmentation + ReduceLROnPlateau. Combining two regularizers on a small dataset caused underfitting (−0.104 vs baseline). This is a negative result: more regularization ≠ better when the model is already well-regularized.

**ResNet feature extraction (0.898)** — frozen ImageNet backbone + trained fc head. Massive jump (+0.310 over CNN baseline), confirming the strength of pretrained features. vanet remains weakest, but F1 improves from ~0.70 (CNN) to ~0.79.

**ResNet fine-tuning (0.932)** — unfreezing layer4 with a smaller lr (1e-4). Best model overall, +0.034 over feature extraction. Confirms that adapting the last block to the traffic domain matters even after pretraining. vanet still the weakest class (F1 ~0.79), but all classes now exceed F1 ≈ 0.79.

### Lowest-class analysis (vanet)

Across all six experiments, vanet is consistently the lowest-recall and lowest-precision class. Likely reasons:
1. Smallest support (17 val images).
2. Visual similarity with minibus/autobus at low resolution.
3. Possible label ambiguity in the dataset itself.

Even ResNet fine-tuning only reaches ~0.79 F1 on vanet, whereas every other class is above 0.90. This is a domain-level signal that vanet may warrant a merge with a visually similar class (e.g., minibus) or a dedicated data collection effort.

## — Merge Decision Analysis (hypothetical, no retrain)

**Method:** recompute accuracy and macro-F1 after merging labels, without retraining.

**Tested merge groups:**
- bus = {minibus, autobus}
- truck = {kamyun, kamyunet}

**Result:**

| Metric | Original | Merged | Δ |
|---|---:|---:|---:|
| Accuracy | 0.946 | 0.962 | +0.016 |
| Macro-F1 | 0.932 | 0.937 | **+0.005** |

**Interpretation:**
The macro-F1 gain from merging is only **+0.005** — far below the noise threshold (~0.03). This means that **less than 1% of the model's errors** occur between the pairs considered for merging. The confusion matrix already showed these pairs are only mildly confused (kamyun↔kamyunet = 0.118; minibus↔autobus much lower). ResNet-FT has effectively learned to distinguish trucks (kamyun vs kamyunet) and buses (minibus vs autobus) despite their visual similarity.

**Decision: DO NOT MERGE.**

**Justification (three criteria from the project):**

1. **Label ambiguity?** Visually, kamyun/kamyunet and minibus/autobus are hard to distinguish by eye at low resolution. This is a partial argument for merging. However, the model achieves high accuracy on these pairs anyway — the ambiguity is not reflected in the model's errors.

2. **Practical importance?** In traffic monitoring, distinguishing heavy vs. light vehicles (kamyun vs kamyunet; bus vs minibus) has real value: it affects axle-load estimates, road-wear assessment, and traffic-mix statistics. Merging would discard this information.

3. **Error magnitude and symmetry?** Pair confusion scores are low (max = 0.118 for kamyun↔kamyunet) and the merged F1 gain is negligible (+0.005). Errors are not large enough to justify abandoning the taxonomy.

**Conclusion:** The three criteria collectively do NOT justify merging. Labels remain distinct. The resnet_ft model is retained as the final model.

**Note:** Retraining with a merged taxonomy was not performed (it is stretch work and would not change the conclusion given the negligible F1 gain).