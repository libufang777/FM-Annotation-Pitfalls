"""Same-image, fixed-prediction label sensitivity; no source file is edited."""
from pathlib import Path
import os, sys, json, hashlib, shutil, time
import numpy as np
from audit_labels import labels, summary, cli
PARSER=cli(__doc__)
PARSER.add_argument('--weights-dir',type=Path,required=True)
PARSER.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1]/'model_runtime')
PARSER.add_argument('--bootstrap',type=int,default=2000)
PARSER.add_argument('--only-model',choices=['YOLO-MSHF','YOLOv8n','YOLOv11n'])
args=PARSER.parse_args()
ROOT=args.v4_root.resolve(); PREP=args.legacy_root.resolve(); OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
WEIGHTS=args.weights_dir.resolve()
os.environ['YOLO_CONFIG_DIR']=str(OUT/'yolo_config')
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_config')
sys.path.insert(0,str(args.runtime.resolve()))
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.ops import xywh2xyxy

def metric(rows, selection=None):
    if selection is None: selection=range(len(rows))
    selected=[rows[i] for i in selection]
    tp,conf,pc,tc=[np.concatenate([r[k] for r in selected]) for k in ('tp','conf','pred_cls','target_cls')]
    result=ap_per_class(tp[:,None],conf,pc,tc,plot=False)
    ap=result[5][:,0]; classes=result[6]
    return {'mAP50':float(ap.mean()),'class_AP50':{str(int(c)):float(v) for c,v in zip(classes,ap)}}

def gt_tensor(path, shape):
    a=labels(path);a=a[np.isin(a[:,0],[0,1])]
    a=a.astype(np.float32)
    _,idx=np.unique(a,axis=0,return_index=True)
    if len(idx)<len(a):a=a[idx]
    t=torch.as_tensor(a,dtype=torch.float32)
    boxes=xywh2xyxy(t[:,1:])*torch.tensor([shape[1],shape[0],shape[1],shape[0]])
    boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,shape[1])
    boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,shape[0])
    return boxes,t[:,0]

class PairedValidator(DetectionValidator):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.paired={k:[] for k in ('v4','manual','v4_agnostic','manual_agnostic')}
        self.raw=[]
    def update_metrics(self,preds,batch):
        super().update_metrics(preds,batch)
        for i,pred in enumerate(preds):
            info=self._prepare_batch(i,batch)
            predn=self._prepare_pred(pred,info)
            name=Path(batch['im_file'][i]).name
            mb,mc=gt_tensor(PREP/'labels_gold'/(Path(name).stem+'.txt'),info['ori_shape'])
            self.raw.append(dict(name=name,predictions=predn.cpu().numpy(),shape=info['ori_shape']))
            for key,b,c in (('v4',info['bbox'],info['cls']),('manual',mb,mc)):
                for agnostic in (False,True):
                    pp=predn.clone(); cc=c.clone()
                    if agnostic: pp[:,5]=0;cc[:]=0
                    tp=self._process_batch(pp,b,cc)[:,0] if len(c) and len(pp) else torch.zeros(len(pp),dtype=torch.bool)
                    self.paired[key+('_agnostic' if agnostic else '')].append(dict(name=name,tp=tp.cpu().numpy(),conf=pp[:,4].cpu().numpy(),pred_cls=pp[:,5].cpu().numpy(),target_cls=cc.cpu().numpy()))

def materialize():
    pairs=json.loads((OUT/'image_correspondences.json').read_text())
    matched=[r for r in pairs if len(r['test'])==1]
    if len(matched)!=300: raise ValueError('Expected 300 unique test matches')
    view=OUT/'paired_view'
    if not view.exists():
        (view/'images/test').mkdir(parents=True)
        (view/'labels/test').mkdir(parents=True)
        for r in matched:
            im=Path(r['legacy_image']); v4=Path(r['test'][0]); lab=v4.parent.parent/'labels'/(v4.stem+'.txt')
            shutil.copy2(im,view/'images/test'/im.name)
            shutil.copy2(lab,view/'labels/test'/(im.stem+'.txt'))
    p=OUT/'paired_v4.yaml'
    p.write_text(f'path: {view.as_posix()}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: Bean_Rust\n  1: Angular_Leaf_Spot\n',encoding='utf-8')
    return p

def main():
    torch.set_num_threads(4)
    data=materialize(); report={}
    for name,fn in (('YOLO-MSHF','YOLO_MSHF_v4_seed0_best.pt'),('YOLOv8n','YOLOv8n_v4_seed0_best.pt'),('YOLOv11n','YOLOv11n_v4_seed0_best.pt')):
        if args.only_model and name!=args.only_model: continue
        cache=OUT/(name+'_paired_stats.npy')
        if cache.exists():
            stored=np.load(cache,allow_pickle=True).item(); paired=stored['paired']; official=stored['official']
            # Reuse fixed predictions, applying the official loader's duplicate-row
            # removal and box clipping to the alternative truth as well.
            matcher=DetectionValidator(args={'plots':False,'device':'cpu','project':str(OUT/'runs'),'name':'matching_only','exist_ok':True})
            for key in ('manual','manual_agnostic'):paired[key]=[]
            for raw in stored['raw']:
                pred=torch.tensor(raw['predictions']);mb,mc=gt_tensor(PREP/'labels_gold'/(Path(raw['name']).stem+'.txt'),raw['shape'])
                for agnostic in (False,True):
                    pp=pred.clone();cc=mc.clone()
                    if agnostic:pp[:,5]=0;cc[:]=0
                    tp=matcher._process_batch(pp,mb,cc)[:,0] if len(cc) and len(pp) else torch.zeros(len(pp),dtype=torch.bool)
                    paired['manual'+('_agnostic' if agnostic else '')].append(dict(name=raw['name'],tp=tp.numpy(),conf=pp[:,4].numpy(),pred_cls=pp[:,5].numpy(),target_cls=cc.numpy()))
            stored['paired']=paired;stored['manual_preprocessing']='official duplicate-row removal and clipping'
            np.save(cache,stored)
        else:
            model=YOLO(str(WEIGHTS/fn))
            if model.names!={0:'Bean_Rust',1:'Angular_Leaf_Spot'}: raise ValueError(model.names)
            instances=[]
            class Capture(PairedValidator):
                def __init__(self,*a,**kw):
                    super().__init__(*a,**kw);instances.append(self)
            m=model.val(validator=Capture,data=str(data),split='test',imgsz=640,batch=8,device='cpu',conf=.001,iou=.5,workers=0,plots=False,verbose=False,half=False,project=str(OUT/'runs'),name=name,exist_ok=True)
            paired=instances[0].paired; official=float(m.box.map50)
            np.save(cache,dict(paired=paired,official=official,raw=instances[0].raw))
        points={k:metric(rows) for k,rows in paired.items()}
        if abs(points['v4']['mAP50']-official)>1e-8: raise ValueError('Reconstructed AP disagrees with official validator')
        if len(paired['v4'])!=300 or [r['name'] for r in paired['v4']]!=[r['name'] for r in paired['manual']]: raise ValueError('Image pairing invalid')
        result={'checkpoint':fn,'sha256':hashlib.sha256((WEIGHTS/fn).read_bytes()).hexdigest(),'images':300,'points':points}
        print(name,json.dumps(points),flush=True)
        rng=np.random.default_rng(20261104); diffs=[];start=time.time()
        for j in range(args.bootstrap):
            sample=rng.integers(0,300,300)
            diffs.append(metric(paired['v4'],sample)['mAP50']-metric(paired['manual'],sample)['mAP50'])
            if (j+1)%200==0: print(name,'bootstrap',j+1,'seconds',round(time.time()-start,1),flush=True)
        result['difference_v4_minus_manual']={'point':points['v4']['mAP50']-points['manual']['mAP50'],'paired_image_bootstrap_ci95':np.percentile(diffs,[2.5,97.5]).tolist(),'iterations':args.bootstrap,'seed':20261104}
        np.save(OUT/(name+'_bootstrap.npy'),np.array(diffs))
        report[name]=result
        (OUT/'evaluation_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    source=json.loads((OUT/'stage1_audit.json').read_text())['source_label_sha256']
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source.items())
    (OUT/'source_integrity.json').write_text(json.dumps({'legacy_label_files_checked':len(source),'all_unchanged':unchanged}),encoding='utf-8')
    if not unchanged: raise ValueError('Source labels changed during audit')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
