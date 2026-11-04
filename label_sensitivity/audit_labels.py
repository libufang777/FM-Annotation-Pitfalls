"""Read-only audit of legacy Gold label provenance and image correspondences."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json
import numpy as np


def cli(description):
    import argparse
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--v4-root',type=Path,required=True,help='Dataset root with test/images and test/labels')
    p.add_argument('--legacy-root',type=Path,required=True,help='Legacy root with images, labels_gold and draft label folders')
    p.add_argument('--out',type=Path,required=True,help='Writable output directory; source data are never edited')
    return p


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

def labels(p):
    rows = []
    for line in p.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            a = list(map(float, line.split()))
            if len(a) != 5: raise ValueError(str(p))
            rows.append(a)
    return np.asarray(rows, dtype=float).reshape(-1,5)

def signature(a):
    return sorted(tuple(np.round(row,6)) for row in a)

def summary(paths):
    arrays = [labels(p) for p in paths]
    counts = Counter(int(r[0]) for a in arrays for r in a)
    two = [a[np.isin(a[:,0], [0,1])] for a in arrays]
    areas = np.concatenate([a[:,3]*a[:,4] for a in two])
    return dict(files=len(paths), class_counts=dict(counts), two_class_boxes=sum(map(len,two)),
                boxes_per_image_mean=float(np.mean(list(map(len,two)))),
                boxes_per_image_median=float(np.median(list(map(len,two)))),
                normalized_area_median=float(np.median(areas)), normalized_area_p95=float(np.percentile(areas,95)))

def main():
    args=cli(__doc__).parse_args(); ROOT=args.v4_root.resolve();PREP=args.legacy_root.resolve();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
    images = sorted(p for p in (PREP/'images').iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg'))
    indexes = {}
    for split in ('test','train','val'):
        ix=defaultdict(list)
        for p in (ROOT/f'{split}/images').iterdir():
            if p.is_file(): ix[md5(p)].append(str(p))
        indexes[split]=ix
    pairs=[]
    for im in images:
        h=md5(im)
        pairs.append(dict(legacy_image=str(im), md5=h, **{s:ix.get(h,[]) for s,ix in indexes.items()}))
    trees=tuple(t for t in ('labels_gold','labels_gold_BEFORE_NMS','labels_gold_cleaned','labels_sam3_draft','labels_sam3draft_FINAL') if (PREP/t).is_dir())
    report={'trees':{},'comparison':{},'images':len(images),'matches':{s:sum(bool(r[s]) for r in pairs) for s in indexes}}
    for tree in trees:
        paths=[PREP/tree/(im.stem+'.txt') for im in images]
        report['trees'][tree]=summary(paths)
        report['trees'][tree]['path']=str(PREP/tree)
    for tree in trees[1:]:
        text_same=sum((PREP/'labels_gold'/(im.stem+'.txt')).read_text(encoding='utf-8-sig').splitlines()==(PREP/tree/(im.stem+'.txt')).read_text(encoding='utf-8-sig').splitlines() for im in images)
        same=sum(signature(labels(PREP/'labels_gold'/(im.stem+'.txt')))==signature(labels(PREP/tree/(im.stem+'.txt'))) for im in images)
        report['comparison'][f'gold_vs_{tree}']={'text_line_identical_images':text_same,'numerically_identical_images':same,'different_images':len(images)-same}
    report['source_label_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for t in trees for p in (PREP/t).glob('*.txt')}
    (OUT/'stage1_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'image_correspondences.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_label_sha256'},indent=2),flush=True)

if __name__=='__main__': main()
