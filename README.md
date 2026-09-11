# FM-RobustAFB

Subgroup calibration and worst-group detection accuracy for acid-fast bacilli (AFB)
detection in Ziehl-Neelsen (ZN) sputum-smear microscopy.

Pooled detection and calibration metrics can look healthy while a small, visually
distinct subgroup of images is badly miscalibrated. This repository contains a
leakage-audited pipeline that measures both at once: worst-group average precision and
subgroup-conditioned candidate calibration, over seven independently trained seeds.

## Headline results

**Present-group equalization (PGE) beats empirical risk minimization (ERM) on worst-group
AP in all 7 seeds** (exact sign test p = 0.0078, Wilcoxon p = 0.0078).

| Reducer | n | Per-seed worst-group AP@0.5 | Mean ± SD |
|---|---|---|---|
| ERM | 7 | 0.40, 0.15, 0.04, 0.07, 0.00†, 0.00, 0.00† | 0.09 ± 0.13 |
| PGE | 7 | 0.42, 0.33, 0.13, 0.41, 0.54, 0.25, 0.63 | 0.39 ± 0.16 |
| DRO-style | 3 | 0.00†, 0.35, 0.00† | 0.12 ± 0.16 |

† collapsed to an all-background solution (0-4 boxes over the whole test set).
PGE never collapsed; ERM reached zero worst-group AP on 3 of 7 seeds. Excluding the two
collapsed ERM runs the comparison is still 5/5 (p = 0.031), so the conclusion does not
depend on how collapsed runs are scored.

**Pooled calibration hides a subgroup failure.** Candidate expected calibration error
(C-ECE) per visual-style group, with the evidence behind each estimate:

| Group | Images | Candidates | Bins occupied | C-ECE | 95% CI |
|---|---|---|---|---|---|
| style 0 | 31 | 221 | 9/15 | 0.15 | [0.09, 0.24] |
| style 1 | 72 | 520 | 9/15 | 0.07 | [0.04, 0.12] |
| style 2 | 5 | 10 | 5/15 | 0.42 | [0.31, 0.61] |
| pooled | 108 | 752 | — | 0.09 | [0.06, 0.13] |

Intervals are percentile bootstrap over test images (2000 resamples), which respects
within-image correlation between candidates. *Images* counts only those contributing at
least one candidate above the detector threshold.

**The gap originates in the detector, not the calibrator.** Stage-wise candidate ECE
after per-signal isotonic calibration:

| Group | Detector | Verifier | Fused |
|---|---|---|---|
| style 0 | 0.094 | 0.102 | 0.148 |
| style 1 | 0.097 | 0.077 | 0.069 |
| style 2 | 0.299 | 0.371 | 0.423 |
| pooled | 0.087 | 0.086 | 0.086 |

Pooled ECE is flat across all three stages while style 2 degrades, and group-conditioned
fusion does not reduce the gap despite being given the group label.

**The verifier confers no measurable benefit at matched recall.** Thresholding the
detector-only and fused scores so each retains the same number of true positives:

| Seed | Matched TP | FP detector-only | FP fused | Change |
|---|---|---|---|---|
| 0 | 838 | 642 | 641 | -0.2% |
| 1 | 435 | 124 | 126 | +1.6% |
| 2 | 177 | 39 | 39 | +0.0% |

Reported as a negative result: the DINOv2 verifier is retained as an audited reference
stage, not as a recommended component.

## Dataset

### What it is

**A Comprehensive Raw Dataset of Ziehl-Neelsen-Stained Sputum Smear Microscopy Images for
*Mycobacterium tuberculosis* Detection** — Yuniarti, Fatichah, Sigit, Arifin and da Costa,
*Data in Brief*, vol. 67, p. 112908, 2026.

Ziehl-Neelsen (ZN) staining is the standard low-cost test for pulmonary tuberculosis: a
sputum smear is stained so that acid-fast bacilli (AFB) retain a red-pink carbolfuchsin
dye against a counterstained background, and a technician scans the slide under a
microscope counting bacilli. It remains a primary screening tool in resource-constrained
settings, and manual reading is slow, fatiguing and variable between readers — which is
what motivates automating bacillus localisation.

The corpus is deliberately **raw**: whole microscope fields as captured, with no cropping,
colour normalisation, focus filtering or curation. That is what makes it suitable for
studying subgroup reliability, because the acquisition variation is still present in the
data rather than pre-processed away.

### Contents

| Property | Value |
|---|---|
| Fields | 1,438 (1,332 after the leakage audit) |
| Resolution | 1632 x 1224 |
| Annotations | 11,447 AFB bounding boxes in YOLO format (10,465 after the audit) |
| Median box size | ~3,973 px², about **0.20%** of a field |
| Camera systems | 2 — Hayear (720), Optilab (718) |
| Background colours | 4 — Yellowish (687), Purplish/Pinkish (570), Bluish (125), Greenish (56) |
| Class | single class (bacillus) |

Targets are extremely small relative to the field, which is why fields are tiled at high
resolution for detection rather than downsampled.

### Acquisition variation

The recorded metadata crosses camera against background colour almost evenly, so the two
axes are close to independent:

| | Bluish | Greenish | Purplish/Pinkish | Yellowish |
|---|---|---|---|---|
| Hayear | 60 | 28 | 287 | 345 |
| Optilab | 65 | 28 | 283 | 342 |

Background colour is severely imbalanced (Greenish and Bluish together are 13% of the
corpus) while camera is balanced. This matters: the calibration failure reported above
tracks the **colour** axis and not the camera axis.

### Obtaining it

The dataset is published with the *Data in Brief* article above and is not redistributed
here. Download it, then point the converter at the extracted folder:

```bash
afb-convert-yolo --dataset-root <path-to-Raw_Sputum_Microscopy_Dataset> --out-dir data/afb
```

The expected layout is `images/{train,val,test}/*.jpg` with matching
`labels/{train,val,test}/*.txt` in YOLO format. An optional
`metadata.csv` with columns `Image_ID, Data_Split, Background_Color, Camera_System`
enables the metadata-axis replication analyses; without it the unsupervised style
grouping still works.

### Leakage audit

The provider splits contain cross-split duplicates, which inflate detection scores if a
field appears in both training and test. Byte-identical files are flagged with SHA-256 and
near-duplicates with a 64-bit perceptual hash at Hamming distance <= 6, which catches
recompressed or reformatted copies of the same field. Flagged pairs are merged into
connected components and one image per component is kept under a train > validation > test
priority, so evaluation-side copies are removed while the training corpus is preserved.

This removes **106 images**, leaving 1,332 fields and 10,465 boxes. Near-duplicate flags
are generated automatically and not exhaustively hand-verified, so these are removed as
*potential* near-duplicates rather than proven ones.

| Split | Images | Boxes | Share | style0 / style1 / style2 |
|---|---|---|---|---|
| train | 1081 | 8499 | 81.2% | 298 / 709 / 74 |
| val | 122 | 882 | 9.2% | 34 / 79 / 9 |
| test | 129 | 1084 | 9.7% | 40 / 82 / 7 |

The style mix is stable across splits (largest share difference 3.4 percentage points), so
the evaluation sets are representative of the corpus. They are small in absolute terms: the
minimum detectable difference in per-candidate correctness between the largest and smallest
style groups is **0.447** at 80% power, so the study is powered only for large subgroup
effects. Bootstrap intervals are reported with every subgroup estimate.

### A note on scope

This is **AFB / bacillus-morphology detection, not species-level tuberculosis diagnosis**.
ZN morphology alone cannot confirm *M. tuberculosis*, and nothing here should be read as a
diagnostic claim.

### Visual-style groups

The corpus carries no patient, slide or acquisition identifiers, so subgroups are derived
from seven low-level features per image (mean R/G/B, mean and standard deviation of luma,
Hasler-Susstrunk colourfulness, Laplacian-variance focus). Feature statistics are fit on
the training split only; k-means with k=3 (ten restarts, fixed seed) is fit on the training
features and validation/test images are assigned to the nearest cluster.

| k | bootstrap ARI | smallest cluster | silhouette |
|---|---|---|---|
| 2 | 0.999 | 74 | 0.592 |
| 3 | 0.951 | 74 | 0.312 |
| 4 | 0.717 | 4 | 0.296 |
| 5 | 0.466 | 4 | 0.225 |

k=3 is the largest cluster count that is both reproducible under resampling (ARI >= 0.9)
and free of degenerate clusters. The partition is not an artefact of the feature space:
style 2 coincides exactly with the recorded *Bluish* background condition (purity
**1.00**, 74/74 training images) while being uncorrelated with the recorded
camera (ARI 0.00005).

## Pipeline

1. **Leakage audit** — exact + perceptual hashing, cross-split duplicate removal.
2. **Detector** — anchor-free FCOS-style head on FPN levels P3-P5 over a ResNet-50
   backbone. Fields capped at a 1600 px long side and tiled into 640x640 crops with 18%
   overlap; predictions merged by cross-tile NMS at IoU 0.5.
3. **Verifier** — each candidate is re-scored by a frozen DINOv2 ViT-B/14 with register
   tokens, adapted by LoRA (rank 8) on the attention and MLP projections. Boxes are
   expanded 2.5x for context and resized to 224x224; a multi-prototype head (8 prototypes
   per class) produces the verifier logit. Supervised only by development-split candidate
   crops. **No gradient reaches the detector.**
4. **Group-robust training** — ERM, present-group equalization, and an EMA-weighted
   DRO-style reducer, all sharing optimizer, schedule, augmentation and a bounded
   cross-style consistency term.
5. **Calibrate-then-fuse** — each signal is calibrated by isotonic regression on the
   development split; a logistic model fuses the two calibrated probabilities, their
   Jensen-Shannon divergence, and a one-hot style-group indicator.

## Installation

```bash
git clone <repository-url> && cd FM-RobustAFB
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python >= 3.10 (the DINOv2 hub module uses `X | Y` type syntax). A CUDA GPU with
>= 8 GB is recommended; training uses gradient checkpointing and peaks near 7.6 GB at
batch size 2 on full-resolution fields.

## Usage

```bash
# 1. convert the YOLO-format corpus to COCO
afb-convert-yolo --dataset-root <path-to-Raw_Sputum_Microscopy_Dataset> --out-dir data/afb

# 2. run the leakage audit and derive visual-style groups
python -m scripts.prepare_groups --config configs/raw_sputum.yaml --k 3

# 3. train and evaluate one reducer on one seed
python -m scripts.run_seed --config configs/raw_sputum.yaml --reducer group_balanced --seed 0

# 4. regenerate every table and statistic from results/
python scripts/build_report.py
```

Each `run_seed` invocation is one self-contained unit of work: one detector, its verifier,
its fusion head and the resulting per-candidate predictions. Completed runs are skipped on
re-invocation and trained detectors are cached on disk by training configuration, so a
sweep can be stopped and resumed without repeating finished work. The reported results are
7 seeds x {ERM, PGE} plus 3 seeds of the DRO-style reducer.

Reproducing the full set:

```bash
for seed in 0 1 2 3 4 5 6; do
  for reducer in erm group_balanced; do
    python -m scripts.run_seed --config configs/raw_sputum.yaml \
        --reducer "$reducer" --seed "$seed"
  done
done
python scripts/build_report.py
```

Pass `--force` to recompute a run that already has a report.

### Compute

One detector is 24 epochs at batch size 2 over 1,081 full-resolution fields.

| | |
|---|---|
| Time per detector | ~1.2 h on a T4, ~6.2 h on a P100 |
| Peak GPU memory | ~7.6 GiB with gradient checkpointing |
| Full sweep | 17 runs, about 21 h on a single T4 |

Training runs on full fields rather than tiles, and the bounded cross-style consistency
term adds two further forward passes per step; without checkpointing the peak footprint is
roughly 21 GiB and will not fit a 16 GB accelerator. Checkpointing is therefore enabled by
default (`AFB_GRAD_CHECKPOINT=0` disables it). It recomputes backbone activations in the
backward pass instead of storing them, which is mathematically identical and leaves trained
weights unchanged, at roughly a 40% slowdown.

On accelerators older than compute capability 7.0 (for example the P100), recent PyTorch
builds ship no compatible kernels and every CUDA operation fails with
`no kernel image is available for execution on the device`. Pin `torch==2.4.1+cu121`
(`torchvision==0.19.1`) there, which still carries `sm_50`-`sm_90`.

## Repository layout

```
fm_robustafb/
  data/         corpus loading, tiling, leakage audit, visual-style grouping
  detector/     FCOS-style detector, backbone, assigner, losses, post-processing
  verifier/     DINOv2 backbone, LoRA adaptation, prototype head
  robust/       reducers (ERM, present-group equalization, Group-DRO) and the
                group-cycling + gradient-accumulation training variant
  fusion/       isotonic calibration, calibrate-then-fuse scoring head
  metrics/      detection AP, candidate calibration (C-ECE, LC-ECE), selective prediction
  engine/       training loops, tiled inference, candidate banks, evaluation,
                per-run driver, activation-memory control
  analysis/     subgroup calibration analyses, dataset statistics, pipeline specification
  experiments/  benchmark, ablation, cross-camera, calibration, baseline studies
configs/        YAML configurations
scripts/        CLI entry points
results/
  runs/         per-run detection and calibration reports (JSON)
  analysis/     computed analyses: counts, bootstrap CIs, stage decomposition, k-selection
  predictions/  per-candidate predictions (.npz) and per-image detections (.pkl)
  figures/      generated figures
tests/          unit tests
```

## Results are reproducible from this repository

`results/predictions/` holds the per-candidate arrays behind every calibration number:
labels, group ids, calibrated detector and verifier probabilities, fused scores, IoUs and
image ids. Every table in this README can be recomputed from them without a GPU:

```bash
python scripts/build_report.py
```

## Metrics

Detection accuracy is summarised by worst-group AP at IoU 0.5, with macro-group AP as a
secondary metric. Calibration uses a candidate-level adaptation of detection expected
calibration error: each candidate is labelled by its best overlap with any ground-truth
box in the same image (IoU >= 0.5 is positive) rather than by one-to-one matching, so the
metric measures candidate-level correctness calibration rather than detection precision.
A localization-aware variant scales credit by IoU. All calibration metrics are conditional
on candidates surviving the 0.05 detector threshold and cannot detect proposal failures
below it.

## Limitations

- Single corpus, single annotation source; no external-dataset replication.
- No patient or slide identifiers, so subgroups are acquisition conditions rather than
  clinical strata.
- The minority style group contributes 10-14 candidates from 5-6 images; its calibration
  estimate is reported with a bootstrap interval and should not be read as a point value.
- Collapse to an all-background solution is a property of this small-batch, three-group
  regime rather than of any single reducer, and every reducer comparison inherits it.
- AFB morphology detection is not species-level tuberculosis diagnosis.

## License

MIT. See `LICENSE`.
