# -*- coding: utf-8 -*-
"""
读取 CSV，只筛选 car_name == ecoron01 的行，
使用 latitude / longitude 列画点并叠加底图。
"""
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import folium

# ==== 配置 ====
CSV_PATH = r"ocartrafficdata.csv"   # ← 改成你的CSV路径
OUTPUT_PNG = "map_points.png"
OUTPUT_HTML = "map_points.html"
POINT_SIZE = 12
TARGET_NAME = "ecoron01"  # 只筛选这个车名
# =============

def robust_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "cp932", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def build_geodf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    # 不区分大小写查找列名
    cols_lower = {c.lower(): c for c in df.columns}
    lat_col = cols_lower.get("latitude")
    lon_col = cols_lower.get("longitude")
    car_col = cols_lower.get("car_name")

    if not lat_col or not lon_col:
        raise ValueError("找不到 latitude/longitude 列")
    if not car_col:
        raise ValueError("找不到 car_name 列")

    # 筛选 car_name
    df = df[df[car_col] == TARGET_NAME].copy()
    if df.empty:
        raise ValueError(f"没有 car_name = {TARGET_NAME} 的数据")

    df["__lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["__lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    mask = df["__lat"].between(-90, 90) & df["__lon"].between(-180, 180)

    gdf = gpd.GeoDataFrame(
        df[mask].reset_index(drop=True),
        geometry=[Point(lon, lat) for lon, lat in zip(df.loc[mask, "__lon"], df.loc[mask, "__lat"])],
        crs="EPSG:4326"
    )
    return gdf

def plot_static_png(gdf: gpd.GeoDataFrame, out_png: str, point_size=12):
    gdf_3857 = gdf.to_crs(epsg=3857)
    xmin, ymin, xmax, ymax = gdf_3857.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    pad_x = max(dx * 0.05, 100)
    pad_y = max(dy * 0.05, 100)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_3857.plot(ax=ax, markersize=point_size, alpha=0.9)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_3857.crs)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_axis_off()
    ax.set_title(f"Points for {TARGET_NAME}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] PNG：{out_png}")

def build_interactive_html(gdf: gpd.GeoDataFrame, out_html: str):
    m = folium.Map(
        location=[float(gdf["__lat"].median()), float(gdf["__lon"].median())],
        zoom_start=12,
        tiles="OpenStreetMap"
    )
    for lat, lon in zip(gdf["__lat"], gdf["__lon"]):
        folium.CircleMarker(location=[lat, lon], radius=4, weight=1, fill=True, fill_opacity=0.9).add_to(m)
    m.fit_bounds([[gdf["__lat"].min(), gdf["__lon"].min()],
                  [gdf["__lat"].max(), gdf["__lon"].max()]])
    m.save(out_html)
    print(f"[OK] HTML：{out_html}")

def main(csv_path):
    df = robust_read_csv(csv_path)
    print(f"[INFO] 读取成功：{len(df)} 行，{len(df.columns)} 列。")
    gdf = build_geodf(df)
    print(f"[INFO] 有效点数：{len(gdf)}")
    plot_static_png(gdf, OUTPUT_PNG, point_size=POINT_SIZE)
    build_interactive_html(gdf, OUTPUT_HTML)

if __name__ == "__main__":
    csv = CSV_PATH
    if len(sys.argv) > 1:
        csv = sys.argv[1]
    if not Path(csv).exists():
        raise FileNotFoundError(f"找不到文件：{csv}")
    main(csv)
