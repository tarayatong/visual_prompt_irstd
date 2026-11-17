"""Layout and draw the paper's qualitative comparison panels."""

import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from visual_prompt.evaluation.comparison import image_metrics

METHODS=['MSHNet+NS-FPN','SCTransNet','MSHNet','DNANet','HDNet','ILNet','SANSA','Ours']
SELECTED=[('SIRST','Misc_229'),('SIRST','Misc_357'),('IRSTD-1k','XDU709'),('IRSTD-1k','XDU745'),('IRSTD-1k','XDU103'),('NUDT-SIRST','001289'),('NUDT-SIRST','001053')]
COLORS={'correct':'#ff3030','missed':'#ffe000','false':'#36e4cf'}
STYLES={'correct':'solid','missed':(0,(2.1,1.8)),'false':(0,(2.1,1.8))}
ZOOM=3.0
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':9,'savefig.facecolor':'white'})

def components(mask):
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    return [dict(id=k,x=float(centers[k,0]),y=float(centers[k,1]),box=stats[k,:4].tolist(),area=int(stats[k,4])) for k in range(1,n)]

def match(gt,pred):
    """Visual-only rule: any response on a GT instance is a detection.

    An overlapping prediction instance is never simultaneously a false alarm.
    Allow multiple fragments per GT and a merged prediction to touch several GTs.
    Official benchmark metrics remain calculated separately, unchanged.
    """
    gn,gl,_,_=cv2.connectedComponentsWithStats(gt.astype(np.uint8),connectivity=8)
    pn,pl,_,_=cv2.connectedComponentsWithStats(pred.astype(np.uint8),connectivity=8)
    overlap=gt & pred;matched={};associated=set()
    if overlap.any():
        pairs=np.unique(np.stack((gl[overlap],pl[overlap]),axis=1),axis=0)
        for g,p in pairs:
            matched.setdefault(int(g)-1,[]).append(int(p)-1)
            associated.add(int(p)-1)
    false=[j for j in range(pn-1) if j not in associated]
    for j in false:assert not np.any(gt & (pl==j+1))
    for i in range(gn-1):assert (i in matched)==bool(np.any(pred & (gl==i+1)))
    return matched,false

def markers(s,method):
    if method in ('Image','GT'):
        values=[(t,'correct',f'T{i+1}') for i,t in enumerate(s['targets'])]
        if method=='Image' and 'F' in s['regions']:
            r=s['regions']['F']
            values.append((dict(x=r['x'],y=r['y'],box=[r['x']-r['size']/2,r['y']-r['size']/2,r['size'],r['size']]),'false',''))
        return values
    rec=s['records'][method]
    values=[(t,'correct' if i in rec['matched'] else 'missed',f'T{i+1}') for i,t in enumerate(s['targets'])]
    values += [(rec['components'][j],'false','') for j in rec['false']]
    return values

def draw_marks(ax,values,scale,labels=False,zoom=False):
    for t,kind,label in values:
        # Thin location circles, capped for large background streaks.
        size=math.hypot(t['box'][2],t['box'][3])/2+2
        r=max(scale*.022,min(size,scale*.070)) if not zoom else min(scale*.42,max(3,size))
        ax.add_patch(Circle((t['x'],t['y']),r,fill=False,color=COLORS[kind],lw=.96,linestyle=STYLES[kind],clip_on=True))
        if labels and label:
            ax.text(t['x']+r+1,t['y']-r-1,label,color=COLORS[kind],fontsize=5.2,clip_on=True)

def setup_image(ax,array,extent=None):
    ax.set_facecolor('black')
    ax.imshow(array,cmap='gray',vmin=0,vmax=255 if array.dtype!=bool else 1,interpolation='nearest')
    if extent:
        x,y,size=extent;ax.set_xlim(x-size/2,x+size/2);ax.set_ylim(y+size/2,y-size/2)
    else:
        h,w=array.shape;size=max(h,w);ax.set_xlim((w-size)/2-.5,(w+size)/2-.5);ax.set_ylim((h+size)/2-.5,(h-size)/2-.5)
    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)

COLS=['Image']+METHODS+['GT']
HEADERS=['Infrared image','MSHNet+\nNS-FPN','SCTransNet','MSHNet','DNANet','HDNet','ILNet','SANSA','CoRe-IR','Ground truth']

def place_insets(s,count):
    """One shared layout for all ten columns, scored against every prediction."""
    if s['sid']=='XDU103':
        # User-selected layout: all three crops in one row flush to the bottom.
        side=.31
        return side,[dict(left=left,top=1-side,placement_source='user specified') for left in (0.,(1-side)/2,1-side)]
    if s['sid']=='001289':
        # User-selected layout: two left-aligned windows stacked at top left.
        side=.30
        return side,[dict(left=0.,top=top,placement_source='user specified') for top in (0.,side+.035)]
    h,w=s['gt'].shape;size=max(h,w);padx=(size-w)/2;pady=(size-h)/2;n=256
    foreground=np.zeros((n,n),np.float32)
    protect=np.zeros((n,n),np.uint8);annotations=np.zeros_like(protect)
    for method in COLS:
        array=s['image'] if method=='Image' else s['gt'] if method=='GT' else s['preds'][method]
        canvas=np.zeros((size,size),np.float32);y0=(size-h)//2;x0=(size-w)//2
        canvas[y0:y0+h,x0:x0+w]=array.astype(np.float32)/(255 if method=='Image' else 1)
        reduced=cv2.resize(canvas,(n,n),interpolation=cv2.INTER_AREA)
        if method!='Image':reduced=cv2.dilate(reduced,np.ones((3,3),np.uint8))
        foreground+=reduced*(0 if method=='Image' else 1)
        for t,kind,label in markers(s,method):
            cx=round((t['x']+padx)/size*n);cy=round((t['y']+pady)/size*n)
            radius=max(size*.022,min(math.hypot(t['box'][2],t['box'][3])/2+2,size*.070))
            cv2.circle(annotations,(cx,cy),math.ceil(radius/size*n)+3,1,-1)
            if label:cv2.circle(protect,(cx,cy),max(9,math.ceil(radius/size*n)+4),1,-1)
    if 'F' in s['regions']:
        r=s['regions']['F'];cv2.circle(protect,(round((r['x']+padx)/size*n),round((r['y']+pady)/size*n)),8,1,-1)
    integrals=[cv2.integral(v.astype(np.float64)) for v in (protect,annotations,foreground)]
    side=.30 if s['sid']=='001289' else .36 if count<=2 else .31
    limit=1-side
    points=np.linspace(0,limit,21);candidates=[]
    for y in points:
        for x in points:
            if not(x==0 or x==limit or y==0 or y==limit):continue
            x0,y0=math.floor(x*n),math.floor(y*n)
            x1,y1=min(n,math.ceil((x+side)*n)),min(n,math.ceil((y+side)*n))
            sums=[float(ii[y1,x1]-ii[y0,x1]-ii[y1,x0]+ii[y0,x0]) for ii in integrals]
            corner_distance=min(x,limit-x)+min(y,limit-y)
            cost=sums[0]*1e12+sums[2]*1e8+sums[1]*1e3+corner_distance+(limit-y)*.00001
            candidates.append((cost,float(x),float(y),sums))
    candidates.sort(key=lambda c:c[0]);beam=[(0.,[])]
    for step in range(count):
        next_beam=[]
        for cost,chosen in beam:
            for c in candidates:
                if any(not(c[1]+side+.01<=old[1] or old[1]+side+.01<=c[1] or c[2]+side+.01<=old[2] or old[2]+side+.01<=c[2]) for old in chosen):continue
                next_beam.append((cost+c[0],chosen+[c]))
        next_beam.sort(key=lambda b:b[0]);beam=next_beam[:30]
        assert beam,'No nonoverlapping inset layout'
    chosen=sorted(beam[0][1],key=lambda c:(c[1],c[2]))
    return side,[dict(left=c[1],top=c[2],covered_gt_protection=c[3][0],covered_annotation=c[3][1],covered_foreground=c[3][2]) for c in chosen]

def render(samples, output: Path):
    n=len(samples);cw=1.72;gap=.055;left=.10;right=.10;head=.56;rh=1.76;foot=.46
    width=left+10*cw+9*gap+right;height=head+n*rh+foot
    fig=plt.figure(figsize=(width,height))
    for j,title in enumerate(HEADERS):
        x=left+j*(cw+gap)
        color='#dce9f2' if j==0 else '#e1efdd' if j==9 else '#f9e4da' if j==8 else '#fff2d1'
        fig.patches.append(Rectangle((x/width,(height-.49)/height),cw/width,.42/height,transform=fig.transFigure,facecolor=color,edgecolor='#b6b6b6',lw=.4))
        fig.text((x+cw/2)/width,(height-.28)/height,title,ha='center',va='center',fontsize=12)
    metadata=[]
    for i,s in enumerate(samples):
        top=height-head-i*rh
        regions=list(s['regions'].items());count=len(regions)
        fraction,placements=place_insets(s,count)
        for j,method in enumerate(COLS):
            x=left+j*(cw+gap)
            array=s['image'] if method=='Image' else s['gt'] if method=='GT' else s['preds'][method]
            ax=fig.add_axes([x/width,(top-cw)/height,cw/width,cw/height])
            setup_image(ax,array);draw_marks(ax,markers(s,method),max(array.shape))
            inset=cw*fraction
            displayed=[];omitted=[];crop_nonzero_pixels={}
            for k,(label,r) in enumerate(regions):
                # Inspect mask pixels before annotation: an empty crop must not
                # become visible merely because a missed-target circle is drawn.
                h,w=array.shape
                x0=max(0,math.ceil(r['x']-r['size']/2));x1=min(w,math.floor(r['x']+r['size']/2)+1)
                y0=max(0,math.ceil(r['y']-r['size']/2));y1=min(h,math.floor(r['y']+r['size']/2)+1)
                nonzero=int(np.count_nonzero(array[y0:y1,x0:x1]))
                crop_nonzero_pixels[label]=nonzero
                if not nonzero:
                    omitted.append(label)
                    continue
                displayed.append(label)
                placement=placements[k]
                ix=x+cw*placement['left'];iy=top-cw*placement['top']-inset
                ia=fig.add_axes([ix/width,iy/height,inset/width,inset/height])
                setup_image(ia,array,(r['x'],r['y'],r['size']))
                local=[m for m in markers(s,method) if abs(m[0]['x']-r['x'])<r['size']/2 and abs(m[0]['y']-r['y'])<r['size']/2]
                draw_marks(ia,local,r['size'],zoom=True)
                for spine in ia.spines.values():spine.set_visible(True);spine.set_linewidth(.65);spine.set_color('#ffffff')
                factor=(inset/r['size'])/(cw/max(array.shape))
                assert abs(factor-ZOOM)<1e-12
            metadata.append(dict(dataset=s['dataset'],sample_id=s['sid'],column=method,regions=s['regions'],magnification=ZOOM,inset_size_fraction=fraction,inset_placements=placements,displayed_regions=displayed,omitted_empty_regions=omitted,crop_nonzero_pixels=crop_nonzero_pixels))
    handles=[Line2D([],[],color=COLORS[k],lw=1.5,linestyle=STYLES[k],marker='o',mfc='none',markersize=7,label=label) for k,label in [('correct','Correct detection'),('missed','Missed target'),('false','False alarm')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09/height),ncol=3,frameon=False,fontsize=12,handlelength=2.8,columnspacing=3)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return metadata
