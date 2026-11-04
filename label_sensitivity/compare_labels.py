"""Label-only statistics and symmetric maximum-cardinality IoU matching."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from audit_labels import labels,summary,cli

def overlap(a,b):
    def xyxy(t):
        return np.c_[t[:,1:3]-t[:,3:5]/2,t[:,1:3]+t[:,3:5]/2]
    a,b=xyxy(a),xyxy(b)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh[:,:,0]*wh[:,:,1]
    aa=np.prod(a[:,2:]-a[:,:2],axis=1); bb=np.prod(b[:,2:]-b[:,:2],axis=1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def main():
    p=cli(__doc__);p.add_argument('--gold350-labels',type=Path,required=True);p.add_argument('--c175-labels',type=Path,required=True)
    args=p.parse_args();ROOT=args.v4_root.resolve();PREP=args.legacy_root.resolve();OUT=args.out.resolve()
    pairs=json.loads((OUT/'image_correspondences.json').read_text())
    vp=[Path(r['test'][0]).parent.parent/'labels'/(Path(r['test'][0]).stem+'.txt') for r in pairs]
    mp=[PREP/'labels_gold'/(Path(r['legacy_image']).stem+'.txt') for r in pairs]
    goldroot=Path(__file__).resolve().parents[1]/'cohorts'
    gold=json.loads((goldroot/'Gold350.json').read_text())['images']
    c=json.loads((goldroot/'C175.json').read_text())['images']
    sets={'v4_same_300':vp,'legacy_manual_same_300':mp,
          'Gold350':[args.gold350_labels/(Path(r['image']).stem+'.txt') for r in gold],
          'C175':[args.c175_labels/(Path(r['image']).stem+'.txt') for r in c]}
    report={'summaries':{name:summary(paths) for name,paths in sets.items()},'matching':{}}
    for agnostic in (False,True):
        matched=total_a=total_b=0; perclass=Counter();confusion=np.zeros((2,2),int)
        for pa,pb in zip(vp,mp):
            a,b=labels(pa),labels(pb);a=a[np.isin(a[:,0],[0,1])];b=b[np.isin(b[:,0],[0,1])]
            total_a+=len(a);total_b+=len(b)
            if not len(a) or not len(b):continue
            iou=overlap(a,b);valid=(iou>=.5)
            if not agnostic: valid &= a[:,0,None]==b[None,:,0]
            ia,ib=linear_sum_assignment(-(valid*1000+iou*valid))
            chosen=valid[ia,ib];ia,ib=ia[chosen],ib[chosen];matched+=len(ia)
            for x,y in zip(ia,ib):
                confusion[int(a[x,0]),int(b[y,0])]+=1
                perclass[int(a[x,0])]+=1
        report['matching']['class_agnostic' if agnostic else 'same_class']={'one_to_one_matched':matched,'v4_boxes':total_a,'manual_boxes':total_b,'fraction_v4_matched':matched/total_a,'fraction_manual_matched':matched/total_b,'class_matrix_v4_rows_manual_columns':confusion.tolist()}
    samegeometry=sameclass=0;manualtotal=0
    for p in mp:
        b=labels(p);b=b[np.isin(b[:,0],[0,1])];manualtotal+=len(b)
        a=labels(PREP/'labels_sam3_draft'/p.name)
        full={tuple(np.round(row,6)) for row in a}; geom={tuple(np.round(row[1:],6)) for row in a}
        sameclass+=sum(tuple(np.round(r,6)) in full for r in b)
        samegeometry+=sum(tuple(np.round(r[1:],6)) in geom for r in b)
    report['manual_inherits_draft']={'manual_disease_boxes':manualtotal,'same_class_and_geometry_to_6_decimals':sameclass,'same_geometry_any_class_to_6_decimals':samegeometry}
    (OUT/'label_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
