#!/usr/bin/env python3
"""Section 6.5 point-level validation: which screen gives the paper's 27 sites?

v11 cell 25 resolves spreadsheet columns by substring, and col("PL") matches
SAMPLE LABEL before the real PL column. Coerced to numeric that is all-NaN, so
the "full laboratory record" screen counted 0 and the cell fell back to a
five-property screen giving 31 sites. This script names every column
explicitly and reports each candidate screen with its Youden index.
"""
import os, time
import numpy as np, pandas as pd, rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, transform as warp_transform

t0=time.time()
# --- data directory resolution -----------------------------------------------
# Was hardcoded to a sandbox path, so this script could not find the rasters on
# any other machine. Mirrors src/validation/fos_means.py: MAKILALA_DIR first,
# then the usual locations, then the working directory.
def _find_data_dir():
    cands = [os.environ.get("MAKILALA_DIR", ""),
             os.path.expanduser("~/Downloads/makilala_data"),
             os.path.expanduser("~/mnt/makilala_data"),
             os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "Mak_BulkUnitWeight.tif")):
            return c
    msg = ["Could not locate makilala_data (no Mak_BulkUnitWeight.tif found).",
           "Tried:"]
    msg += ["  " + x for x in cands if x]
    msg += ["Set it explicitly:  MAKILALA_DIR=/path/to/makilala_data python "
            + os.path.basename(__file__)]
    raise SystemExit(chr(10).join(msg))


INPUT_DIR = _find_data_dir()
GAMMA_W, ALPHA_MIN, GRID_CRS, REF_K = 9.81, 10, "EPSG:32651", 'gamma'
THETA=0.90
FILES={'slope':"Slope_PRSTM51N.tif",'depth':"Soil Depth.tif",'pga':"PGA 2019 EQ Events.tif",
       'gamma':"Mak_BulkUnitWeight.tif",'water':"Mak_WaterContent.tif",'pi':"Mak_PlasticityIndex.tif",
       'fines':"Mak_Fines.tif",'sand':"Mak_Sand.tif"}

def estimate_phi(pi,sand,fines):
    phi=np.full_like(pi,30.0,dtype=np.float64)
    v=(pi>5)&np.isfinite(pi); phi[v]=35.0-0.1*pi[v]
    no=~v&np.isfinite(sand)
    phi[no&(sand>85)&(fines<10)]=33.0
    phi[no&(sand>70)&(fines>=10)&(fines<20)]=31.0
    phi[no&(fines>=20)]=28.0
    return np.clip(phi,20.0,38.0)
def estimate_c(fines,sand):
    c=np.full_like(fines,5.0,dtype=np.float64); v=np.isfinite(fines)&np.isfinite(sand)
    c[v&(fines<5)]=1.0; c[v&(fines>=5)&(fines<12)]=3.0
    c[v&(fines>=12)&(fines<20)]=6.0; c[v&(fines>=20)]=10.0
    return c
def estimate_m(w):
    m=np.full_like(w,0.5,dtype=np.float64); v=np.isfinite(w)&(w>0)
    m[v&(w<20)]=0.2; m[v&(w>=20)&(w<35)]=0.4
    m[v&(w>=35)&(w<50)]=0.6; m[v&(w>=50)]=0.8
    return m
def compute_FoS(c,phid,ad,m,g,z,kh):
    phi,a=np.radians(phid),np.radians(ad)
    R=c+(g-m*GAMMA_W)*z*np.cos(a)**2*np.tan(phi)
    D=g*z*np.sin(a)*np.cos(a)+kh*g*z*np.cos(a)**2
    return np.where(D>0,R/D,np.nan)

ras,prof={},{}
for k,fn in FILES.items():
    with rasterio.open(os.path.join(INPUT_DIR,fn)) as s:
        d=s.read(1).astype(np.float64)
        if s.nodata is not None: d[d==s.nodata]=np.nan
        ras[k],prof[k]=d,s.profile.copy()
ref_shape=ras[REF_K].shape; ref_transform=prof[REF_K]['transform']; ref_crs=CRS.from_string(GRID_CRS)
al={}
for k,d in ras.items():
    if d.shape==ref_shape and prof[k]['transform'].almost_equals(ref_transform,precision=1e-6):
        al[k]=d; continue
    dst=np.full(ref_shape,np.nan)
    reproject(d,dst,src_transform=prof[k]['transform'],src_crs=CRS.from_user_input(prof[k]['crs']),
              dst_transform=ref_transform,dst_crs=ref_crs,resampling=Resampling.bilinear,
              src_nodata=np.nan,dst_nodata=np.nan)
    al[k]=dst
del ras
alpha=np.clip(al['slope'],0.1,80.0); gamma=np.clip(al['gamma'],GAMMA_W+0.1,30.0)
phi=estimate_phi(al['pi'],al['sand'],al['fines']); c=estimate_c(al['fines'],al['sand'])
m=estimate_m(al['water']); z_raw=al['depth']
pga=al['pga']; pmax=np.nanmax(pga[np.isfinite(pga)&(pga>0)])
pg=pga/981.0 if pmax>100 else (pga/9.81 if pmax>2 else pga); k_h=np.clip(0.5*pg,0.0,1.0)
z5=np.clip(z_raw,0.1,5.0)
valid=(np.isfinite(alpha)&(alpha>=ALPHA_MIN)&np.isfinite(z_raw)&(z_raw>0)&np.isfinite(gamma)&(gamma>GAMMA_W)
       &np.isfinite(phi)&(phi>0)&np.isfinite(c)&(c>=0)&np.isfinite(m)&(m>=0)&(m<=1))
FOS=np.full(ref_shape,np.nan)
FOS[valid]=compute_FoS((c+10.0)[valid],phi[valid],alpha[valid],m[valid],gamma[valid],z5[valid],k_h[valid])
SCORED=valid&np.isfinite(FOS)
print(f"SCORED {int(SCORED.sum()):,}  (published 7,104,440)   {time.time()-t0:.0f}s",flush=True)

# ---- spreadsheet, EXPLICIT column names ------------------------------------
df=pd.read_excel(os.path.join(INPUT_DIR,"Makilala Geotechnical Data.xlsx"),sheet_name=0,header=1)
df.columns=[str(x).strip() for x in df.columns]
LAT,LON,TYPE="LATITUDE","LONGITUDE","SITE TYPE"
W,BUW,DUW="WATER CONTENT (%)","BULK UNIT WEIGHT (kN/m3)","DRY UNIT WEIGHT (kN/m3)"
GRAV,SAND,FINES,LL,PL,PI="Gravel %","Sand %","Silt/Clay %","LL","PL","PI"
NINE=[W,BUW,DUW,GRAV,SAND,FINES,LL,PL,PI]; FIVE=[PI,SAND,FINES,W,BUW]
df=df[pd.to_numeric(df[LAT],errors='coerce').notna()].copy()
for cc in NINE+[LAT,LON]: df[cc]=pd.to_numeric(df[cc],errors='coerce')
is_ls=df[TYPE].astype(str).str.upper().str.startswith('LANDSLIDE').values
print(f"rows with coords {len(df)}   landslide/non {is_ls.sum()}/{(~is_ls).sum()}")
print(f"PL non-null (explicit column): {df[PL].notna().sum()}   "
      f"[v11 col('PL') hits SAMPLE LABEL -> 0]")

xs,ys=warp_transform("EPSG:4326",ref_crs,df[LON].tolist(),df[LAT].tolist())
inv=~ref_transform
R=np.array([int(np.floor((inv*(x,y))[1])) for x,y in zip(xs,ys)])
C=np.array([int(np.floor((inv*(x,y))[0])) for x,y in zip(xs,ys)])
ok=(R>=0)&(R<ref_shape[0])&(C>=0)&(C<ref_shape[1])
in_dom=np.zeros(len(df),bool); in_dom[ok]=SCORED[R[ok],C[ok]]
print(f"in scored domain: {in_dom.sum()}   (paper 36)")

full9=df[NINE].notna().all(axis=1).values
full5=df[FIVE].notna().all(axis=1).values
print(f"\n{'screen':<42}{'n':>5}{'LS':>5}")
for lab,sel in [("in domain",in_dom),("in domain & complete on five",in_dom&full5),
                ("in domain & complete on NINE (full lab)",in_dom&full9),
                ("complete on nine, anywhere",full9)]:
    print(f"  {lab:<40}{sel.sum():>5}{(sel&is_ls).sum():>5}")

def score(sel,label):
    i=np.where(sel)[0]
    if len(i)==0: return
    fmap=FOS[R[i],C[i]]
    cm=estimate_c(df[FINES].values[i],df[SAND].values[i])+10.0
    pm=estimate_phi(df[PI].values[i],df[SAND].values[i],df[FINES].values[i])
    mm=estimate_m(df[W].values[i])
    gm=np.clip(df[BUW].values[i],GAMMA_W+0.1,30.0)
    fmeas=compute_FoS(cm,pm,alpha[R[i],C[i]],mm,gm,z5[R[i],C[i]],k_h[R[i],C[i]])
    y=is_ls[i]
    for nm,f in (("map",fmap),("measured",fmeas)):
        pred=f<THETA; good=np.isfinite(f)
        tp=int((pred&y&good).sum()); fn=int((~pred&y&good).sum())
        fp=int((pred&~y&good).sum()); tn=int((~pred&~y&good).sum())
        hit=tp/max(tp+fn,1); far=fp/max(fp+tn,1)
        print(f"    {label:<34}{nm:<9}n={good.sum():<3} TP{tp:>3} FN{fn:>3} FP{fp:>3} TN{tn:>3}"
              f"  hit {hit:5.1%} FAR {far:5.1%}  J {hit-far:.3f}")
    both=np.isfinite(fmap)&np.isfinite(fmeas)
    if both.sum()>1:
        d=fmeas[both]-fmap[both]
        diff=int(((fmap[both]<THETA)!=(fmeas[both]<THETA)).sum())
        print(f"    {'':<34}{'both':<9}n={both.sum():<3} dFoS mean {d.mean():+.3f} sd {d.std(ddof=1):.3f}"
              f"  differ {diff}/{both.sum()}")

print(f"\nscoring (threshold {THETA}):")
score(in_dom&full5,"complete on five")
score(in_dom&full9,"complete on nine")
print(f"\npaper reports: 27 sites, J 0.097, 3 of 27 differ, dFoS +0.037 sd 0.137")
print(f"v11 cell 25  : 31 sites, J 0.127, 4 of 31 differ, dFoS +0.036 sd 0.138")
print(f"done {time.time()-t0:.0f}s")
