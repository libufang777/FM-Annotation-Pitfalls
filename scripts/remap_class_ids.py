"""Copy YOLO labels while remapping historical LabelImg IDs to project IDs.

Historical mapping: 0 insect -> 2; 1 bean rust -> 0; 2 angular -> 1.
Never edits source labels. Use only for the historical pre-correction cohort.
"""

import argparse
from collections import Counter
from pathlib import Path


MAPPING = {0: 2, 1: 0, 2: 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.source.is_dir()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    counts = Counter()
    files = sorted(p for p in args.source.glob("*.txt") if p.name != "classes.txt")
    for source in files:
        lines = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed YOLO line in {source}")
            old = int(parts[0])
            if old not in MAPPING:
                raise ValueError(f"Unexpected class {old} in {source}")
            parts[0] = str(MAPPING[old])
            counts[MAPPING[old]] += 1
            lines.append(" ".join(parts))
        (args.out / source.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (args.out / "classes.txt").write_text("Bean_Rust\nAngular_Leaf_Spot\nInsect_Feeding_Hole\n", encoding="utf-8")
    print(f"Copied and remapped {len(files)} label files; class counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
