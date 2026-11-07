"""Fixed region/distance diagnostics and labelled, comparable prediction panels."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw


def error_sums(pred,target,mask):
    active=np.asarray(mask,bool)
    e=np.asarray(pred,np.float64)[active]-np.asarray(target,np.float64)[active]
    if not np.isfinite(e).all():
        raise ValueError("nonfinite evaluation error")
    return dict(pixels=int(active.sum()),squared_error_m2=float(np.square(e).sum()),absolute_error_m=float(np.abs(e).sum()))


def summary(sums):
    n=sums["pixels"]
    return dict(sums,rmse_m=math.sqrt(sums["squared_error_m2"]/n) if n else None,
                mae_m=sums["absolute_error_m"]/n if n else None)


def regions(target_mask,input_mask,task,bins):
    masks={"all":target_mask}
    if task["pipeline"]["protocol"]=="dense_sr":
        return masks  # LR block observations are NOT aligned HR point measurements
    masks.update(observed=target_mask & input_mask,missing=target_mask & ~input_mask)
    h,w=target_mask.shape
    fraction=task["pipeline"].get("hole_fraction",0)
    if fraction:
        hh=max(1,int(round(h*fraction**.5)));ww=max(1,int(round(w*fraction**.5)))
        y,x=(h-hh)//2,(w-ww)//2
        hole=np.zeros((h,w),bool);hole[y:y+hh,x:x+ww]=True
        if input_mask[hole].any():
            raise ValueError("central-hole protocol differs from original degradation")
        masks.update(central_hole=target_mask & hole,outside_central_hole=target_mask & ~hole)
    distance=distance_transform_edt(~input_mask)
    for left,right in zip(bins,bins[1:]):
        masks[f"distance_{left}_{right}_px"]=target_mask & (distance>=left)&(distance<right)
    return masks


def correction_diagnostics(pred,initial,target,mask):
    p,i,t=(np.asarray(a,np.float64)[mask] for a in (pred,initial,target))
    correction=p-i
    after,before=np.abs(p-t),np.abs(i-t)
    return dict(correction_mean_m=float(correction.mean()),correction_abs_mean_m=float(np.abs(correction).mean()),
                correction_abs_max_m=float(np.abs(correction).max()),
                improved_pixels=int((after<before).sum()),worsened_pixels=int((after>before).sum()),
                unchanged_pixels=int((after==before).sum()),pixels=len(p))


def texture_diagnostic(pred,target,mask,rgb):
    """Counts, not a perceptual proof: RGB edges on nearly flat GT depth."""
    gray=rgb.astype(np.float64).mean(-1)/255
    eligible=false=0
    safe=np.where(mask,target,0)
    for axis in (0,1):
        a=np.take(mask,range(mask.shape[axis]-1),axis=axis)
        b=np.take(mask,range(1,mask.shape[axis]),axis=axis)
        active=a&b&(np.abs(np.diff(safe,axis=axis))<.01)&(np.abs(np.diff(gray,axis=axis))>=.05)
        eligible+=int(active.sum())
        false+=int((active&(np.abs(np.diff(pred,axis=axis))>=.05)).sum())
    return dict(rgb_edge_flat_gt_pairs=eligible,false_depth_edge_pairs=false,
                false_depth_edge_fraction=false/eligible if eligible else None)


def panel(path,rgb,observations,input_mask,target,valid,predictions,title):
    """Fixed [0,10] m depth colors; fixed [0,.5] m error colors, no per-image scaling."""
    h,w=target.shape
    display_depth=observations if observations.shape==target.shape else np.asarray(
        Image.fromarray(observations).resize((w,h),Image.Resampling.NEAREST))
    display_mask=input_mask if input_mask.shape==target.shape else np.asarray(
        Image.fromarray(input_mask.astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST)).astype(bool)
    def depth_color(a,m=None):
        value=np.clip(a/10,0,1)
        color=np.stack([value,np.sqrt(value),1-value],-1)
        color=(255*color).astype(np.uint8)
        if m is not None:color[~m]=0
        return color
    tiles=[("Registered RGB",rgb),("Input; black = invalid",depth_color(display_depth,display_mask)),
           ("GT depth [0,10] m",depth_color(target,valid)),("GT validity",np.repeat((valid*255).astype(np.uint8)[...,None],3,-1))]
    for name,pred in predictions.items():
        tiles.append((name+" [0,10] m",depth_color(pred)))
        error=np.clip(np.abs(pred-target)/.5,0,1)
        color=np.stack([error,np.zeros_like(error),1-error],-1)
        color[~valid]=0
        tiles.append((name+" |error| [0,.5] m",(255*color).astype(np.uint8)))
    tw,th,cols,label=320,240,4,25
    rows=math.ceil(len(tiles)/cols)
    canvas=Image.new("RGB",(tw*cols,(th+label)*rows+45),"white")
    draw=ImageDraw.Draw(canvas);draw.text((8,8),title,fill="black")
    for k,(name,array) in enumerate(tiles):
        x,y=(k%cols)*tw,45+(k//cols)*(th+label)
        draw.text((x+4,y+4),name,fill="black")
        canvas.paste(Image.fromarray(array).resize((tw,th),Image.Resampling.NEAREST),(x,y+label))
    with open(path,"xb") as f:canvas.save(f,format="PNG")
