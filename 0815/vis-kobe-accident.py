# -*- coding: utf-8 -*-
import os, re, sys, math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import osmnx as ox

# ======= 配置 =======
CSV_PATH = r'C:\Users\79152\Downloads\kotsujiko2017-2024 (2).csv'  # ← 改成你的CSV路径
OUT_DIR  = r'./outputs'
LAT_IDX  = 15   # 第16列（索引15）= 纬度(Y)
LON_IDX  = 16   # 第17列（索引16）= 经度(X)
NETWORK_TYPE = 'drive'
HEX_GRID_SIZE = 60
BBOX_EXPAND_METERS = 1500  # 在数据外再扩 1.5 km
# ====================

def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)

def try_read_table(path):
    encodings = ['utf-8-sig','cp932','shift_jis','utf-8']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine='python')
        except: pass
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep='\t', engine='python')
        except: pass
    raise RuntimeError('读取失败，请手动指定编码/分隔符')

def to_float_safe(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace('，', ',')
    if s.count(',')==1 and s.count('.')==0: s = s.replace(',', '.')
    else: s = s.replace(',', '')
    try: return float(s)
    except: return np.nan

def main():
    ensure_outdir(OUT_DIR)
    df = try_read_table(CSV_PATH)

    # 按列序号读取经纬度
    df['_lat'] = df.iloc[:, LAT_IDX].map(to_float_safe)
    df['_lon'] = df.iloc[:, LON_IDX].map(to_float_safe)

    # 丢掉无效坐标
    df = df.dropna(subset=['_lat', '_lon']).copy()
    print(df)
    bad = (~df['_lon'].between(-180, 180)) | (~df['_lat'].between(-90, 90))
    if bad.any():
        print(f'⚠️ 警告：发现 {bad.sum()} 行经纬度超范围，将被丢弃。')
        df = df[~bad].copy()
    if df.empty:
        raise ValueError('没有有效经纬度数据')

    # → GeoDataFrame (WGS84)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['_lon'], df['_lat'])],
        crs='EPSG:4326'
    )

    # 投影到米制（3857）求包络并扩展
    gdf_3857 = gdf.to_crs(3857)
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    expand = BBOX_EXPAND_METERS
    minx2, miny2, maxx2, maxy2 = minx - expand, miny - expand, maxx + expand, maxy + expand

    # 用矩形 Polygon 作为 bbox（在3857），再整体投影回4326
    bbox_poly_3857 = gpd.GeoSeries([box(minx2, miny2, maxx2, maxy2)], crs=3857)
    bbox_poly_4326 = bbox_poly_3857.to_crs(4326)
    west, south, east, north = bbox_poly_4326.total_bounds  # 直接拿到 west,south,east,north

    # 下载 OSM 路网
    print('Downloading OSM network ...')
    ox.settings.use_cache = True
    ox.settings.log_console = False
    G = ox.graph_from_bbox(north=north, south=south, east=east, west=west,
                           network_type=NETWORK_TYPE, simplify=True)
    _, gdf_edges = ox.graph_to_gdfs(G)

    # 统一到3857作图
    edges_3857 = gdf_edges.to_crs(3857)
    pts_3857 = gdf_3857

    # 图1：路网 + 事故点
    fig, ax = plt.subplots(figsize=(10,10), dpi=150)
    edges_3857.plot(ax=ax, linewidth=0.6, color='#9ca3af')
    pts_3857.plot(ax=ax, markersize=6, color='#ef4444', alpha=0.6)
    ax.set_title('Accident Points (col#16=lat, col#17=lon) over OSM', fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'points_xy_osm.png'))
    plt.close(fig)

    # 图2：路网 + Hexbin 频度
    fig2, ax2 = plt.subplots(figsize=(10,10), dpi=150)
    edges_3857.plot(ax=ax2, linewidth=0.6, color='#9ca3af')
    hb = ax2.hexbin(pts_3857.geometry.x, pts_3857.geometry.y,
                    gridsize=HEX_GRID_SIZE, bins='log', alpha=0.85)
    cb = fig2.colorbar(hb, ax=ax2)
    cb.set_label('log(Count)')
    ax2.set_title('Accident Frequency (Hexbin) over OSM', fontsize=14)
    ax2.set_axis_off()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, 'hexbin_xy_osm.png'))
    plt.close(fig2)

    print('✅ 可视化完成，输出文件：')
    print(os.path.abspath(os.path.join(OUT_DIR, 'points_xy_osm.png')))
    print(os.path.abspath(os.path.join(OUT_DIR, 'hexbin_xy_osm.png')))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('运行出错：', repr(e))
        sys.exit(1)
