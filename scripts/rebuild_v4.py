"""Verify or materialize the species-clean v4 split from the archived v2 split.

The blacklist and both manifests are the retained records of the actual v4
selection. The default mode only checks them; --materialize makes a new tree.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
MANIFESTS = PACKAGE / "manifests"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, help="v2 directory containing train/val/test images and labels")
    parser.add_argument("--out", type=Path, help="New v4 output root; required with --materialize")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    v2 = json.loads((MANIFESTS / "clean_split_v2_manifest_relative.json").read_text(encoding="utf-8"))
    v4 = json.loads((MANIFESTS / "clean_split_v4_manifest_relative.json").read_text(encoding="utf-8"))
    with (MANIFESTS / "candidate_cucumber_removal.csv").open(encoding="utf-8", newline="") as stream:
        blacklist = {row["stem"] for row in csv.DictReader(stream)}
    assert len(blacklist) == 2366
    selected = {}
    for split in ("train", "val", "test"):
        entries = [row for row in v2[split] if split != "train" or Path(row["name"]).stem not in blacklist]
        actual = {row["name"] for row in entries}
        frozen = {row["name"] for row in v4[split]}
        assert actual == frozen, f"v4 {split} disagrees with retained manifest"
        selected[split] = sorted(actual)
    assert [len(selected[s]) for s in ("train", "val", "test")] == [8313, 1656, 828]
    print("Verified v4 split: train=8313, val=1656, test=828; removed=2366")
    if not args.materialize:
        return
    if not args.v2_root or not args.v2_root.is_dir() or not args.out:
        parser.error("--materialize requires an existing --v2-root and a new --out")
    if args.out.exists():
        raise FileExistsError(args.out)
    for split, names in selected.items():
        for name in names:
            source_image = args.v2_root / split / "images" / name
            source_label = args.v2_root / split / "labels" / (Path(name).stem + ".txt")
            if not source_image.is_file() or not source_label.is_file():
                raise FileNotFoundError(f"Missing v2 pair: {source_image} / {source_label}")
    for split, names in selected.items():
        image_out = args.out / split / "images"
        label_out = args.out / split / "labels"
        image_out.mkdir(parents=True)
        label_out.mkdir(parents=True)
        for name in names:
            shutil.copy2(args.v2_root / split / "images" / name, image_out / name)
            label_name = Path(name).stem + ".txt"
            shutil.copy2(args.v2_root / split / "labels" / label_name, label_out / label_name)
    print(f"Materialized new v4 tree at {args.out}")


if __name__ == "__main__":
    main()
