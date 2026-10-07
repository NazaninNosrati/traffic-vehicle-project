# Traffic Vehicle Classification

Reliable traffic-vehicle classification with CNNs and transfer learning (ResNet18).

## Status

- [x] Data audit complete (duplicates, label conflicts, mislabel suspects)
- [x] Cleaned splits + validation created (80/20 stratified, seed 42)
- [x] CNN baseline trained
- [x] Ablations complete (augmentation, dropout, pooling, weight decay, scheduler)
- [x] Balanced batch sampler under simulated imbalance
- [x] CE vs BCE comparison (equal-budget tuning)
- [x] ResNet-18 feature extraction
- [x] ResNet-18 fine-tuning
- [x] Confusion matrix + pair confusion analysis
- [x] Merge decision (no merge)
- [x] `unclean` / `neysan` unseen-class analysis
- [x] `needs_review` threshold selected from validation (0.70)
- [x] `predict.py` with JSON output
- [ ] Final test evaluation (run once when test set is released)
- [x] README fully reproducible by another person

---

## Setup

```bash
# Clone
git clone <repo-url>
cd mini-project-02

# Environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install
pip install -r requirements.txt
data/
├── raw/
│   └── dataset/
│       ├── train/             
│       ├── unclean/         
│       └── test/               
└── processed/
    ├── train/                  
    └── val/              

Data audit summary
Total scanned: 4,360 images across train, test, unclean.

Exact duplicates (MD5): 24 groups; 1 label conflict (Nissan pickup labeled vanet in train and neysan in unclean). Train copy removed.

Perceptual duplicates (average hash, hash_size=16): 46 candidate groups; 8 label-conflict candidates. All false positives reviewed by eye; none removed without confirmation.

Mislabel suspects (ResNet-18 embeddings + centroid distance): 15 farthest images per class reviewed. 3 images moved to correct class, 3 excluded (belong to no class: van, police car, hearse).

Cleaned split: train=1,539, val=385, test=400.

See EXPERIMENTS.md for the full audit log, including every exclusion decision.
Seed: 42 (Python random, NumPy, PyTorch, and DataLoader shuffle via torch.Generator().manual_seed(42)).

Validation split: StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) on cleaned train.

Test set: Frozen. Its file list + MD5 are stored in data/processed/test_files.csv. Test is used once for final evaluation.

Transforms:

Train (baseline): Resize((128,128)) → ToTensor → Normalize(ImageNet). No augmentation.

Train (ResNet): Resize((224,224)) → ToTensor → Normalize(ImageNet).

Val / test: same as train, without augmentation.

Configs: every experiment writes a row to results/experiments.csv with the full config (optimizer, lr, epochs, batch_size, augmentation, dropout, pool type, weight_decay, scheduler, sampler, class_weights, n_train).

Checkpoints: checkpoints/<experiment>.pt contains model_state_dict, epoch, val_macro_f1, img_size, plus metadata (architecture, class_names, class_to_idx, threshold, seed).



## Project structure
mini-project-02/
├── src/
│   ├── dataset.py          # scanning, hashing, splits, loaders, transforms
│   ├── model.py            # CNNBaseline, TLResNet
│   ├── train.py            # fit, train_one_epoch, make_scheduler
│   ├── evaluate.py         # predict_loader, summarize, rank_confused_pairs,
│   │                       # evaluate_on_test, evaluate_unseen
│   ├── experiment.py       # run_experiment, save_result
│   ├── sampler.py          # BalancedBatchSampler
│   └── utils.py            # set_seed, helpers
├── notebooks/
│   └── analysis.ipynb      # full pipeline
├── results/
│   ├── experiments.csv     # every experiment's metrics + config
│   ├── comparison_table.csv
│   ├── pair_confusion.csv
│   ├── cm_count.png
│   ├── cm_norm.png
│   ├── manual_relabel_log.csv
│   └── ...
├── checkpoints/            # best model per experiment (NOT in Git)
├── logs/                   # per-epoch training logs (NOT in Git)
├── data/                   # (NOT in Git)
├── EXPERIMENTS.md
├── README.md
└── requirements.txt

## How to reproduce the final result
# 1. Prepare cleaned splits (from raw)
python -m src.dataset --prepare

# 2. Train ResNet-18 with fine-tuning (final model)
python -m src.experiment --name resnet_ft --architecture resnet18 \
    --trainable fc,layer4 --lr-head 1e-3 --lr-layer4 1e-4 \
    --epochs 20 --batch-size 32 --img-size 224 --seed 42

# 3. Evaluate on validation
python -m src.evaluate --checkpoint checkpoints/resnet_ft.pt --split val

# 4. (Once, when test is released) evaluate on test
python -m src.evaluate --checkpoint checkpoints/resnet_ft.pt \
    --test-root data/raw/dataset/test

## Results summary
Validation comparison table
Experiment	Macro P	Macro R	Macro F1	Lowest-Recall	Lowest-Precision
CNNbaseline |	0.612 | 0.600 |	0.588 |	vanet |	vanet
Balancedbatches |	0.532 |	0.540 |	0.534 |	vanet |	vanet
CrossEntropy\BCE|	0.490 |	0.538 |	0.501 |	vanet |	vanet
Bestregularized\scheduled |	0.478 |	0.516 |	0.484 |	vanet |	vanet
ResNet_feature
-extraction |	0.902 |	0.895 |	0.898 |	vanet |	vanet
ResNet-fine-tuning |0.936 |	0.930 |	0.932|	vanet | vanet

Key findings
Transfer learning dominates. ResNet-18 fine-tuning (0.932) beats CNN baseline (0.588) by +0.344 macro-F1.

Architecture matters, but less than pretraining. ResNet from scratch reaches 0.827 (+0.262 over CNN); pretrained feature extraction adds another +0.071.

Regularization was not the bottleneck. All regularization ablations (dropout, augmentation, weight decay, scheduler) reduced F1. The CNN baseline had a small gap (0.094).

BCE underperformed CE at lr=1e-3, but equal-budget tuning reverses the conclusion: BCE at lr=3e-3 reaches 0.631, beating CE's best (0.596).

BalancedBatchSampler is essential under imbalance. ambulance recall collapsed to 0.04 without sampler; recovered to 0.78 with sampler (+0.129 macro-F1).

vanet is the hardest class across every model (17 val images; visual overlap with minibus/savari).

No merge is justified. Merging adds only +0.005 macro-F1 — below noise (~0.03).

The model is overconfident. Median confidence on correct = 1.000; mean on wrong = 0.848.

Unseen-class detection is limited. On neysan, mean confidence = 0.80 — the model forces a known class.

Confusion matrix
Count-based and row-normalized confusion matrices: results/cm_count.png, results/cm_norm.png. Pair-confusion ranking: results/pair_confusion.csv.

Misclassified image analysis
12 highest-confidence errors analyzed in EXPERIMENTS.md. Dominant patterns: kamyun ↔ kamyunet (4), taxi → savari (3). All errors have confidence > 0.85 — overconfidence, not capacity, is the core limitation.

Domain challenge: ambulance vs vanet
Many ambulances are built on Mazda pickup (vanet) chassis; only emergency equipment distinguishes them. Domain-level ambiguity, not label noise.

Stretch work attempted
ResNet-18 from scratch for 32×32 input (conv1: 3×3 stride=1, maxpool removed). F1 = 0.827.

Equal-budget tuning for CE vs BCE across two learning rates.

Merge decision analysis (hypothetical, no retrain).

Not attempted: temperature scaling, OOD detection, Grad-CAM, FastAPI endpoint, unit tests.

Limitations & notes
Small val support (~50 images/class) means small metric differences (< 0.03) are within noise.

Test set is frozen and released separately.

Overconfident softmax limits the reliability of needs_review for unseen classes.

unclean was never used for training — analysis only.

Reproducibility: all runs use seed 42 and write full config to results/experiments.csv.
