"""Read-only MD5 and 8x8 pHash screening across image directories.

pHash matches the revision audit: 32x32 grayscale Lanczos resize, two-axis
DCT, upper-left 8x8 median threshold, then Hamming distance. A near match
is a screening candidate, not proof of identical image provenance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def images(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    coeff = dct(dct(pixels, axis=0), axis=1)[:8, :8]
    bits = (coeff > np.median(coeff)).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True, help="Directory of images to screen")
    parser.add_argument("--reference", type=Path, required=True, help="Reference image directory")
    parser.add_argument("--out", type=Path, required=True, help="New JSON report path")
    args = parser.parse_args()
    assert args.query.is_dir() and args.reference.is_dir()
    if args.out.exists():
        raise FileExistsError(args.out)
    query = images(args.query)
    reference = images(args.reference)
    if not query or not reference:
        raise ValueError("Both directories must contain images")
    ref = [(p.name, md5(p), phash(p)) for p in reference]
    rows = []
    for path in query:
        checksum = md5(path)
        code = phash(path)
        nearest = min(((code ^ item[2]).bit_count(), item[0]) for item in ref)
        exact = [item[0] for item in ref if checksum == item[1]]
        rows.append({"query": path.name, "md5": checksum, "nearest_pHash_distance": nearest[0],
                     "nearest_reference": nearest[1], "md5_identical_references": exact})
    report = {"query_count": len(query), "reference_count": len(reference),
              "md5_overlap_images": sum(bool(r["md5_identical_references"]) for r in rows),
              "pHash_le_5": sum(r["nearest_pHash_distance"] <= 5 for r in rows),
              "pHash_le_10": sum(r["nearest_pHash_distance"] <= 10 for r in rows),
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
