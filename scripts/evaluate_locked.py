"""Reproduce the seed-0 YOLO-MSHF v4 test and fixed Gold350 mAP50.

The package contains no dataset images or labels. The caller supplies the
reconstructed v4 tree and the cleaned New Gold 402 archive. Gold350 is
materialized from its frozen image+MD5 manifest, with insect boxes excluded.
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((PACKAGE / "configs/locked_eval.json").read_text(encoding="utf-8"))


def checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_yaml(root, destination):
    text = (f"path: {root.resolve().as_posix()}\n"
            "train: images/test\nval: images/test\ntest: images/test\n"
            "names:\n  0: Bean_Rust\n  1: Angular_Leaf_Spot\n")
    destination.write_text(text, encoding="utf-8")


def check_v4(v4_root):
    manifest = json.loads((PACKAGE / "manifests/clean_split_v4_manifest_relative.json").read_text(encoding="utf-8"))
    pair_manifest = json.loads((PACKAGE / "manifests/v4_test_pairs_md5.json").read_text(encoding="utf-8"))
    expected = {row["name"] for row in manifest["test"]}
    actual = {p.name for p in (v4_root / "test/images").iterdir() if p.is_file()}
    if actual != expected or len(actual) != CONFIG["v4_test_images"]:
        raise ValueError("v4 test image roster differs from the retained v4 manifest")
    if pair_manifest["count"] != len(pair_manifest["pairs"]) or {r["image"] for r in pair_manifest["pairs"]} != actual:
        raise ValueError("v4 test pair-hash roster differs from the retained v4 manifest")
    labels = v4_root / "test/labels"
    for row in pair_manifest["pairs"]:
        name = row["image"]
        image = v4_root / "test/images" / name
        label = labels / row["label"]
        if not label.is_file():
            raise FileNotFoundError(label)
        if label.name != Path(name).stem + ".txt" or checksum(image) != row["image_md5"] or checksum(label) != row["label_md5"]:
            raise ValueError(f"v4 test image/label hash differs from frozen pair: {name}")
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip() and int(line.split()[0]) not in (0, 1):
                raise ValueError(f"v4 test has an unexpected class: {label}")


def build_gold350(gold_root, destination):
    manifest = json.loads((PACKAGE / "cohorts/Gold350.json").read_text(encoding="utf-8"))
    assert manifest["count"] == len(manifest["images"]) == CONFIG["gold350_images"]
    images_out = destination / "images/test"
    labels_out = destination / "labels/test"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)
    counts = Counter()
    for entry in manifest["images"]:
        name = entry["image"]
        image = gold_root / "images" / name
        label = gold_root / "labels" / (Path(name).stem + ".txt")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing Gold image/label: {image} / {label}")
        if checksum(image) != entry["md5"]:
            raise ValueError(f"Gold image MD5 does not match frozen manifest: {name}")
        shutil.copy2(image, images_out / name)
        lines = []
        for raw in label.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            cls = int(raw.split()[0])
            if cls in (0, 1):
                lines.append(raw)
                counts[cls] += 1
            elif cls != 2:
                raise ValueError(f"Unexpected Gold class {cls}: {label}")
        (labels_out / label.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if counts != Counter({0: 1319, 1: 751}):
        raise ValueError(f"Gold350 class counts disagree with paper: {counts}")
    return counts


def evaluate(model, data_yaml, cohort, out):
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=CONFIG["image_size"],
        batch=CONFIG["batch"], device=CONFIG["device"],
        conf=CONFIG["confidence"], iou=CONFIG["nms_iou"],
        workers=CONFIG["workers"], half=CONFIG["half"],
        plots=False, verbose=False, project=str(out / "ultralytics_runs"),
        name=cohort, exist_ok=False,
    )
    return float(metrics.results_dict["metrics/mAP50(B)"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, required=True, help="clean_split_v4 with test/images and test/labels")
    parser.add_argument("--gold-root", type=Path, required=True, help="new_gold402 with images and cleaned three-class labels")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--weights", type=Path, default=PACKAGE / "weights/YOLO_MSHF_v4_seed0_best.pt")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assert args.gold_root.is_dir() and args.weights.is_file()
    check_v4(args.v4_root)
    args.out.mkdir(parents=True)
    gold_eval = args.out / "eval_data_gold350"
    counts = build_gold350(args.gold_root, gold_eval)
    v4_view = args.out / "v4_view"
    (v4_view / "images").mkdir(parents=True)
    # The v4 source tree remains untouched. A symlink keeps one locked YAML layout.
    # If symlinks are unavailable on Windows, copy only the 828 test pairs.
    for kind in ("images", "labels"):
        target = v4_view / kind / "test"
        source = args.v4_root / "test" / kind
        try:
            target.symlink_to(source.resolve(), target_is_directory=True)
        except OSError:
            shutil.copytree(source, target)
    yaml_v4 = args.out / "v4_test.yaml"
    yaml_gold = args.out / "gold350.yaml"
    write_yaml(v4_view, yaml_v4)
    write_yaml(gold_eval, yaml_gold)
    sys.path.insert(0, str(PACKAGE / "model_runtime"))
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    model = YOLO(str(args.weights))
    if len(model.names) != 2:
        raise ValueError(f"Expected two-class weight, got {model.names}")
    results = {
        "weights": str(args.weights.resolve()),
        "config": CONFIG,
        "gold350_class_counts": dict(counts),
        "v4_test_mAP50": evaluate(model, yaml_v4, "v4_test", args.out),
        "Gold350_mAP50": evaluate(model, yaml_gold, "Gold350", args.out),
    }
    results["matches_three_decimal_paper"] = (
        round(results["v4_test_mAP50"], 3) == 0.761 and
        round(results["Gold350_mAP50"], 3) == 0.250
    )
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not results["matches_three_decimal_paper"]:
        raise SystemExit("Reevaluation differs from three-decimal paper values; inspect results.json")


if __name__ == "__main__":
    main()
