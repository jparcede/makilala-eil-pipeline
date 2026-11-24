#!/usr/bin/env python3
"""Co-location of the 25 landslide field sites with the mapped inventory."""
import os,numpy as np,pandas as pd,rasterio,geopandas as gpd
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform as warp_transform
INPUT_DIR=os.environ.get("MAKILALA_DIR",os.path.expanduser("~/mnt/makilala_data"))
with rasterio.open(os.path.join(INPUT_DIR,"Mak_BulkUnitWeight.tif")) as s:
    ref_shape=(s.height,s.width); ref_transform=s.transform
ref_crs=CRS.from_string("EPSG:32651"); PX=abs(ref_transform.a)
g=gpd.read_file(os.path.join(INPUT_DIR,"LSI_North Cotabato.shp"))
g=g.set_crs("EPSG:32651",allow_override=True)          # RELABEL, never reproject
OBS=rasterize(((geom,1) for geom in g.geometry if geom is not None),
              out_shape=ref_shape,transform=ref_transform,fill=0,
              all_touched=False,dtype='uint8').astype(bool)
print(f"inventory polygons {len(g)};  centre-rasterised pixels {int(OBS.sum()):,}")
df=pd.read_excel(os.path.join(INPUT_DIR,"Makilala Geotechnical Data.xlsx"),sheet_name=0,header=1)
df.columns=[str(c).strip() for c in df.columns]
df=df[pd.to_numeric(df["LATITUDE"],errors='coerce').notna()].copy()
for c in ("LATITUDE","LONGITUDE"): df[c]=pd.to_numeric(df[c],errors='coerce')
is_ls=df["SITE TYPE"].astype(str).str.upper().str.startswith("LANDSLIDE").values
xs,ys=warp_transform("EPSG:4326",ref_crs,df["LONGITUDE"].tolist(),df["LATITUDE"].tolist())
inv=~ref_transform
R=np.array([int(np.floor((inv*(x,y))[1])) for x,y in zip(xs,ys)])
C=np.array([int(np.floor((inv*(x,y))[0])) for x,y in zip(xs,ys)])
def near_px(r,c,max_m=600.0):
    rad=int(np.ceil(max_m/PX))
    r0,r1=max(0,r-rad),min(OBS.shape[0],r+rad+1); c0,c1=max(0,c-rad),min(OBS.shape[1],c+rad+1)
    sub=OBS[r0:r1,c0:c1]
    if not sub.any(): return np.inf
    rs,cs=np.nonzero(sub)
    return float((np.hypot((rs+r0)-r,(cs+c0)-c)*PX).min())
# A: distance to nearest centre-rasterised pixel centre (what v11 does)
dA=np.array([near_px(r,c) for r,c in zip(R,C)])[is_ls]
# B: true planimetric distance from the point to the polygon geometry
pts=gpd.GeoSeries(gpd.points_from_xy(np.array(xs)[is_ls],np.array(ys)[is_ls]),crs=ref_crs)
uni=g.geometry.union_all()
dB=pts.distance(uni).values
for lab,d in (("A: nearest rasterised pixel centre",dA),("B: distance to polygon geometry",dB)):
    print(f"\n{lab}")
    print(f"   inside / d=0      {int((d==0).sum())}      (paper 3)")
    print(f"   within 25 m       {int((d<=25).sum())}      (paper 15)")
    print(f"   within 50 m       {int((d<=50).sum())}      (paper 19)")
    print(f"   median            {np.median(d):.1f} m   (paper 25)")
