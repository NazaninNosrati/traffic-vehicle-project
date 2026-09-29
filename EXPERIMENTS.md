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
