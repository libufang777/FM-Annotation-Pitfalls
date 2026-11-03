# Auditing the Hidden Pitfalls of Foundation-Model-Assisted Annotation in Agricultural Vision: A Third-Party Holdout Stress Test on Bean Leaf Disease Detection

This repository contains the reviewer-requested reproducibility materials for
the revised bean-disease audit. It contains no dataset photographs,
dataset labels, SAM200 individual ratings, chat records, or internal notes.
Code and fixed cohort manifests are provided here; checkpoints are available in the [v1.0 release](https://github.com/libufang777/FM-Annotation-Pitfalls/releases/tag/v1.0).

## What this reproduces

With the seed-0 YOLO-MSHF v4 `best.pt`, the locked CPU protocol should give
mAP50 **0.761** on `clean_split_v4/test` (828 images) and **0.250** on the fixed
Gold 350 cohort. These are three-decimal publication values. The evaluation
script saves the unrounded values in `results.json` and raises an error if the
three-decimal values differ. A Python 3.10, PyTorch 2.7.1 CPU recheck from a
separate, data-staged directory gave 0.761474 on v4 test and 0.249564 on
Gold 350, matching the publication values. This check used a virtual
environment with the specified packages and inherited the locally installed
PyTorch runtime; it was not a fully independent installation of PyTorch.
The script does not retrain a model. Checkpoints are release assets, not part
of Git history.

## Required external data

Obtain these separately; neither is included in this repository:

1. `clean_split_v4/` with `test/images/` and `test/labels/`, using the frozen
   `manifests/clean_split_v4_manifest_relative.json` roster and
   `manifests/v4_test_pairs_md5.json` image/label checksums. The v4 test labels
   must use the two-class mapping: `0=Bean_Rust`, `1=Angular_Leaf_Spot`.
2. The **revised, lesion-level** New Gold 402 archive, with `images/` and
   `labels/` directly under the specified root. The cleaned Gold labels use
   `0=Bean_Rust`, `1=Angular_Leaf_Spot`, `2=Insect_damage`. The script excludes
   class 2 from evaluation and selects Gold 350 via
   `cohorts/Gold350.json` (image names and MD5 checksums). The 350 selected
   images must yield 1,319 rust and 751 angular boxes after filtering.

Dataset DOI: [10.57760/sciencedb.38554](https://doi.org/10.57760/sciencedb.38554).
**Version 2 is public. Version 3, which adds the fixed evaluation cohort
lists and an updated README, is under repository review.** Reproduction does
not require waiting for Version 3: use the public Version 2 data and the
fixed manifests in this repository:

| Cohort | Images | Manifest |
|---|---:|---|
| Gold 350 | 350 | [Gold350.json](cohorts/Gold350.json) |
| Gold 402 | 402 | [Gold402.json](cohorts/Gold402.json) |
| GoldA | 175 | [GoldA.json](cohorts/GoldA.json) |
| GoldB | 175 | [GoldB.json](cohorts/GoldB.json) |
| C175 | 175 | [C175.json](cohorts/C175.json) |

Download `MSHF_clean_split_v4.zip` and `MSHF_NewGold402.zip` from Version 2;
`MSHF_v4_safe_labels.zip` contains the SAM-completion label supplement.
The active two-disease labels in `MSHF_NewGold402.zip` contain **2,070 boxes:
1,319 rust and 751 angular leaf spot**. Select the 350-image primary cohort
with `cohorts/Gold350.json`, retaining all listed images, including the 24
with no two-disease boxes. The archive contains earlier documentation;
the current paper and this repository define the fixed evaluation protocol.
The Version 3 submission updates the documentation and supplies the same
fixed cohort lists separately.
The New Gold photographs originate from the third-party Marconi Lab corpus;
the provenance screening found no MD5-identical image in v4 and disclosed
one altered GoldA/training photograph. This is not a claim of complete
acquisition independence.

## Environment

The retained project runtime is in `model_runtime/ultralytics/` because the
YOLO-MSHF checkpoint uses custom model classes absent from stock Ultralytics.
It identifies as Ultralytics 8.3.63 and carries the upstream AGPL-3.0 license
in `model_runtime/LICENSE_AGPL-3.0.txt`. The code is distributed
under AGPL-3.0; the dataset's license is specified separately by ScienceDB.
The retained evaluation used Python 3.10, PyTorch
2.7.1, torchvision 0.22.1, image size 640, batch 8, `device=cpu`, candidate
confidence 0.001, NMS IoU 0.5, and half precision off.

In a Python 3.10 virtual environment, install the dependencies in
`requirements.txt`. The locked CPU test was run in a separate directory using
the supplied manifests, data, and checkpoint; verify package versions before
comparing results.

## One-command publication-value check

Download the checkpoint release assets into `weights/` first. From this
directory, after the data and Python dependencies are available:

```powershell
python scripts/evaluate_locked.py --v4-root "data/clean_split_v4" --gold-root "data/new_gold402" --weights "weights/YOLO_MSHF_v4_seed0_best.pt" --out "results/mshf_locked_check"
```

Use a new `--out` path; the script refuses to overwrite an existing run. It
checks every v4 test image and label against its MD5, verifies every selected Gold image against its MD5,
filters the Gold insect labels, copies only the evaluation pairs into its new
output directory (or links the v4 test pairs when permitted), and runs the
locked CPU evaluation. It never changes the source data. The expected
three-decimal results are `0.761` and `0.250`; keep the unrounded values and
the complete log when comparing runs.

The five cohort manifests (`Gold350`, `Gold402`, `GoldA`, `GoldB`, `C175`) carry
frozen filenames and MD5 values. `scripts/rebuild_v4.py` audits the v2-to-v4
manifest transition (8,313/1,656/828 and 2,366 removed images) without
materializing data by default. `scripts/audit_md5_phash.py` and
`scripts/remap_class_ids.py` expose the image-screening and historical
class-ID remapping steps. See `--help` for each script.

## Training configurations and checkpoint mapping

The checkpoint release contains 32 files mapped in
`checkpoint_manifest_SHA256.json`: baseline v4 best-validation weights for
three backbones and three seeds; three v4_safe seed-0 weights; NR-3/5/7;
coverage A/B/C final-epoch weights for three seeds plus B/C best-validation
sensitivity weights; the no-bean_relabelled best checkpoint; and the v2
pre-species-cleaning best checkpoint. Coverage A best-validation sensitivity
uses the same three YOLO-MSHF v4 baseline best weights. Final-epoch `last.pt`
is the **primary** coverage result; `best.pt` is sensitivity only. Earlier
single-run coverage checkpoints are not included in this release; they were
not retained with sufficient provenance to match the reported run.

`configs/training_run_args/<checkpoint-stem>.yaml` is a copy of the corresponding
run's actual `args.yaml`, retaining numeric hyperparameters while normalizing
machine-specific path fields. Checkpoint copies likewise differ from the
originals only in path-valued metadata; model tensors are byte-for-byte equal
after loading. Original checkpoint SHA256 values are retained in the authors'
internal provenance register, and publishable copy SHA256 values are in the
public manifest. The custom YOLO-MSHF architecture is in
`configs/models/yolo_mshf_v2_blpa.yaml`.

`configs/data_clean_split_v4_original.yaml` and
`configs/data_clean_split_v4_safe_original.yaml` record the dataset mapping
with portable paths. Resolve these paths to the downloaded ScienceDB data
before retraining. The locked *evaluation* uses `configs/locked_eval.json`
and `scripts/evaluate_locked.py`, not the validation defaults in the training
run files.

## Boundaries of this package

The Git repository contains code, configurations and manifests;
checkpoint files are staged as release assets, not committed to Git history.
It does not contain photographs, label files, raw SAM200 rater sheets,
internal notes, or unrelated experiments. Reproduction still requires the
ScienceDB data, the separately released checkpoint files, and a compatible
Python environment. The repository and release attachments are intended to be publicly accessible for review and reproduction.

## Checkpoint SHA256 values

The following files are distributed as release attachments, not tracked in Git.
The `paper` column identifies the corresponding manuscript table; the exact
evaluation command for every checkpoint/cohort pair is in
`checkpoint_manifest_SHA256.json`. Verify an attachment against its SHA256
before evaluation.

| Checkpoint | SHA256 | Paper table | Role |
|---|---|---|---|
| `YOLO_MSHF_v4_seed0_best.pt` | `a2cb9fd2b5e281d5c22bd1bd84d35a1f157cf63ea87e3df59974d156197d01e9` | tab:mainresults; tab:seeds | best-validation |
| `YOLO_MSHF_v4_seed1_best.pt` | `c9163cf1ff3affd606b6ca3a392cb5433c7ffb914c98921a8d4d2440d2ced4b7` | tab:seeds | best-validation |
| `YOLO_MSHF_v4_seed2_best.pt` | `1240bcedc59066ec6e630a7814b90508511a5b1e631204aa761508bd3126dc93` | tab:seeds | best-validation |
| `YOLOv8n_v4_seed0_best.pt` | `35242241e5d33064e055d2174b5b56874bf572814720a0f51c8281dd3ec01b4e` | tab:mainresults; tab:seeds | best-validation |
| `YOLOv8n_v4_seed1_best.pt` | `83b73f8924fdf4e1647769080f8b096ad160b7eb4d7ee6c64fb8036cf923c224` | tab:seeds | best-validation |
| `YOLOv8n_v4_seed2_best.pt` | `fdcb7f4c8b9681d52fc08fbeb038824b99b21ce7c7d017f4d00b9538ba79f64f` | tab:seeds | best-validation |
| `YOLOv11n_v4_seed0_best.pt` | `4f07f46c6eb70c5f386ea2d6d6e1789980e962109aab3e4c7825fdffd94d60ad` | tab:mainresults; tab:seeds | best-validation |
| `YOLOv11n_v4_seed1_best.pt` | `c5763124da311c24dc8683e679dff86b2f008362f82b1ab353464acb1f32a5d2` | tab:seeds | best-validation |
| `YOLOv11n_v4_seed2_best.pt` | `29b09d273d72dedb4eedbf568d990c0d02d1c715374363b7356f99952d5a1806` | tab:seeds | best-validation |
| `YOLO_MSHF_v4_safe_seed0_best.pt` | `111c6480b6107df07219b745f54f3b5248a50a5934e68b80a32af7c43b8e887d` | tab:mainresults; tab:sam | best-validation |
| `YOLOv8n_v4_safe_seed0_best.pt` | `b3cf6d122461f1e8fb4ab9ffdb5d8c4de129a34c266e71067277840cc3468073` | tab:mainresults; tab:sam | best-validation |
| `YOLOv11n_v4_safe_seed0_best.pt` | `dbbb49dfe0d358e3380fb815b32327f17d800bf855478fbca32845cbc7514ae0` | tab:mainresults; tab:sam | best-validation |
| `NR3_seed0_best.pt` | `8f88594d15fcfe7c8def0073511bcdf1fb31b886a5a7888f4422d60d1f0decee` | tab:nr | best-validation |
| `NR5_seed0_best.pt` | `9735147823d6a004f95654b05c8d95c6053f64c66c5691c7240f8b5b2ec1b3f0` | tab:nr | best-validation |
| `NR7_seed0_best.pt` | `6701f3073c2b0dcffb047effb36d98caa77cffa7074ebb70a8901991d0f3f5e8` | tab:nr | best-validation |
| `coverage_A0_last.pt` | `a92034d6e91089d8c11a2517a50b7adceb4dcf603f0030230a62b31b405d34fc` | tab:coverage3seed | primary-final-epoch |
| `coverage_A1_last.pt` | `88a89e541578d0a9b2fc70599e31256806970bb0dbeb6604537de9584d89a191` | tab:coverage3seed | primary-final-epoch |
| `coverage_A2_last.pt` | `84186fb13a43c70fca45f6790d0f8385ac4348bf2bbdc2377d1103dfceb91bc3` | tab:coverage3seed | primary-final-epoch |
| `coverage_B0_last.pt` | `f130146d73dfc749169c2ad143b81bcdf23787d9b4f9ebde174981d5b0938d40` | tab:coverage3seed | primary-final-epoch |
| `coverage_B0_best.pt` | `479489055722797097f5e05a0940903b270c5147e73dbf6d771c8378a8d6d4db` | tab:coverage3seed sensitivity | best-validation-sensitivity |
| `coverage_B1_last.pt` | `6afc8426b5b9af60801c7f9dc8efaae0323ef142201212a81f00d871300292b3` | tab:coverage3seed | primary-final-epoch |
| `coverage_B1_best.pt` | `c23712a2e374cd4cd2a667ff0ed70f2c05fbb9469fc6af0869e0f82611e0176d` | tab:coverage3seed sensitivity | best-validation-sensitivity |
| `coverage_B2_last.pt` | `71158fd76d563ea7c05ef7241c1e9922d7d9dccb6d851f0f24aed7d8678067bb` | tab:coverage3seed | primary-final-epoch |
| `coverage_B2_best.pt` | `10dd8ac4b856eb8fefe36af92808f60302f45caaa64ab1dbc8469d95f3dea71c` | tab:coverage3seed sensitivity | best-validation-sensitivity |
| `coverage_C0_last.pt` | `4127af02882c83c9e11865dcf8195ebe801d6f59624ae6d8f3f3ad0ad08b55e9` | tab:coverage3seed | primary-final-epoch |
| `coverage_C0_best.pt` | `b99384316d3badaf8effe799fa65b2d909d59b82f027b09ab4c85d25f71d6115` | tab:coverage3seed sensitivity | best-validation-sensitivity |
| `coverage_C1_last.pt` | `ca6391b2bde618064f691c4f2a238b391d277e605aa91fef4feb92c3c81c9047` | tab:coverage3seed | primary-final-epoch |
| `coverage_C1_best.pt` | `c0d1fd4d619c25138ce7b3206d51845969471fcbf449562c987f76e229989736` | tab:coverage3seed sensitivity | best-validation-sensitivity |
| `coverage_C2_last.pt` | `0211228eb578c5ed374c9398ca83f4cd9d7fd128cdf113daf4d81643d581bea2` | tab:coverage3seed | primary-final-epoch |
| `coverage_C2_best.pt` | `342bdfd2e93b994f62e65a3c9719b18c9d2e7419baee5836cb99410e09169a94` | tab:coverage3seed sensitivity | best-validation-sensitivity |
| `no_bean_relabelled_seed0_best.pt` | `9bab5345c025373290368bb41b7e633ea1f6fa4fc225cabbd1961ef3164d265e` | Section 5.4; reviewer 2-2.1 | best-validation |
| `v2_pre_species_cleaning_seed0_best.pt` | `60e828537aa6c226fb6d1fd198f3a3ef7a5c4206cec38ef8196c811ed0618dbb` | Section 6.3; Gate 6 | best-validation |

