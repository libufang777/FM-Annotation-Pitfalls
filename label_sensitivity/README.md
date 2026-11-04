# Same-image label-sensitivity check (Appendix A.6)

This check holds images and predictions fixed while changing ground-truth labels.
It is not an estimate of the fraction of the new Gold gap caused by annotation.
The 300 images were selected by the legacy SAM procedure, not randomly sampled.

## Inputs

Use the repository environment and `model_runtime` described in the main README.
Obtain v4 images/labels from ScienceDB (DOI: 10.57760/sciencedb.38554) and the three
seed-0 v4 weights from Release v1.0. The earlier author-refined label set must be
requested from the authors. The first-version Gold archive contains SAM drafts,
not this refined set. Do not substitute those drafts for the refined labels.
`author_refined_label_sha256.json` identifies the exact raw refined files.

Arrange your local legacy inputs as `images/`, `labels_gold/`,
`labels_sam3draft_FINAL/` and `labels_sam3_draft/`. The latter contains the earlier
pre-FINAL drafts used only for the inheritance comparison. No images or labels
are distributed in this repository.

## Run from the repository root

Replace the example relative data paths with your own local dataset locations.

```bash
python label_sensitivity/audit_labels.py --v4-root ./datasets/v4 --legacy-root ./datasets/legacy_gold300 --out ./results/label_sensitivity
python label_sensitivity/compare_labels.py --v4-root ./datasets/v4 --legacy-root ./datasets/legacy_gold300 --out ./results/label_sensitivity --gold350-labels ./datasets/holdout/labels --c175-labels ./datasets/v4/val/labels
python label_sensitivity/evaluate_paired.py --v4-root ./datasets/v4 --legacy-root ./datasets/legacy_gold300 --out ./results/label_sensitivity --weights-dir ./weights --bootstrap 2000
```

The audit must run first; it maps the images by MD5 and records local input hashes.
The evaluator creates copies in its output directory; it never alters source
labels. Run in an empty output directory when changing inputs or model weights:
cached prediction files are only valid for the exact original inputs and weights.
Use `--only-model YOLO-MSHF` to evaluate just that checkpoint. All evaluation uses
CPU, 640-pixel input, conf=0.001, NMS IoU=0.5, FP32 and two disease classes.
AP50 matching uses IoU=0.5. Insect boxes are excluded. The official duplicate-row
removal and clipping are applied to the alternative truth too. Class-agnostic
scores merge disease IDs and rematch the same post-NMS predictions.

## Frozen results

`evaluation_results.json` records checkpoint SHA256, per-class and merged-class
scores, and 2,000 paired image-bootstrap intervals (seed 20261104). These intervals
do not measure training-seed uncertainty. YOLO-MSHF scores are 0.748463 with v4
labels and 0.332934 with refined labels on these 300 images. The label counts are
2,017 versus 4,702 raw disease rows; eight duplicate rust rows are removed at
loading, leaving 4,694 evaluated refined boxes. Their means are 6.72, 15.67 raw,
and 15.65 after duplicate removal. Neither label set is independently adjudicated
pathology truth. The refined legacy protocol differs from the Gold 350 protocol,
so this sensitivity check cannot decompose the Gold gap.

Code follows the repository license. Data and checkpoints retain the terms stated
in the main README and their respective source releases.
