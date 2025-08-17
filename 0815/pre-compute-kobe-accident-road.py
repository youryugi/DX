# precompute_edge_accidents.py
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, LineString, box
from shapely.strtree import STRtree
import warnings
import os

# ========= 配置 =========
CSV_PATH   = r"kotsujiko2017-2024.csv"   # 事故CSV路径
LAT_COL    = "緯度（北緯）"   # 第24列（纬度）
LON_COL    = "経度（東経）"   # 第25列（经度）
ENCODING   = "shift_jis"      # 读取编码，失败会回退 utf-8

# 下载OSM路网覆盖范围：以事故点外包矩形向外扩
PADDING_M  = 3000             # 米

# 缓存文件
GRAPH_CACHE_PATH = "road_graph_bbox.graphml"       # 下载后的原始路网缓存

# 输出
OUT_EDGES_CSV      = "edges_accidents.csv"
OUT_GRAPH_GRAPHML  = "road_graph_with_accidents.graphml"
# =======================

def read_csv_jp(path, enc="shift_jis"):
    try:
        return pd.read_csv(path, encoding=enc)
    except Exception:
        return pd.read_csv(path, encoding="utf-8")

def build_edge_gdf(G_proj):
    recs = []
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            p1 = Point((G_proj.nodes[u]["x"], G_proj.nodes[u]["y"]))
            p2 = Point((G_proj.nodes[v]["x"], G_proj.nodes[v]["y"]))
            geom = LineString([p1, p2])
        length = float(data.get("length", geom.length))
        recs.append({"u": u, "v": v, "key": k, "geometry": geom, "length_m": length})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=G_proj.graph["crs"])

def download_graph_bbox_from_points(gdf_pts, padding_m=2000):
    pts_3857 = gdf_pts.to_crs(3857)
    xmin, ymin, xmax, ymax = pts_3857.total_bounds
    xmin -= padding_m; ymin -= padding_m; xmax += padding_m; ymax += padding_m
    bbox_geo = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs=3857).to_crs(4326)
    lon_min, lat_min, lon_max, lat_max = bbox_geo.total_bounds
    G = ox.graph_from_bbox(lat_max, lat_min, lon_max, lon_min, network_type="drive")
    return G

def main():
    # 1) 读取事故数据
    df = read_csv_jp(CSV_PATH, ENCODING)
    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise KeyError(f"未找到列：{LAT_COL} 或 {LON_COL}")
    df = df[[LAT_COL, LON_COL]].dropna()
    df = df[df[LAT_COL].between(30.0, 46.5) & df[LON_COL].between(128.0, 146.5)]
    if df.empty:
        raise ValueError("无有效事故点。")

    # 2) 转为点（WGS84）
    gdf_pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df[LON_COL].to_numpy(), df[LAT_COL].to_numpy())],
        crs=4326
    )

    # 3) 路网加载（优先用缓存）
    if os.path.exists(GRAPH_CACHE_PATH):
        print(f"✅ 发现缓存路网：{GRAPH_CACHE_PATH}，直接加载…")
        G = ox.load_graphml(GRAPH_CACHE_PATH)
    else:
        print("↻ 下载OSM路网…")
        G = download_graph_bbox_from_points(gdf_pts, PADDING_M)
        print("xiazaichenggong")
        ox.save_graphml(G, GRAPH_CACHE_PATH)
        print(f"✅ 已保存路网缓存：{GRAPH_CACHE_PATH}")

    print("duqu chenggong")

    # 4) 投影到米制
    G_proj = ox.project_graph(G)
    crs_proj = G_proj.graph["crs"]

    # 5) 投影事故点
    gdf_pts_proj = gdf_pts.to_crs(crs_proj)

    # 6) 边表（确保索引是 0..N-1 连续）
    edges_gdf = build_edge_gdf(G_proj).reset_index(drop=True)
    edges_gdf["edge_idx"] = edges_gdf.index  # 保存一个稳定的索引列

    # 7) 用空间最近连接把每个事故点匹配到最近边
    #   - 需要 GeoPandas >= 0.10（一般都有）
    joined = gpd.sjoin_nearest(
        gdf_pts_proj[["geometry"]],  # 左：事故点
        edges_gdf[["edge_idx", "geometry"]],  # 右：边，只带索引列和几何
        how="left",
        distance_col="nearest_dist"
    )

    # 8) 统计每条边命中的事故数量
    counts = joined["edge_idx"].value_counts().rename_axis("edge_idx").reset_index(name="accident_count")

    # 9) 回填到 edges_gdf
    edges_gdf = edges_gdf.merge(counts, on="edge_idx", how="left")
    edges_gdf["accident_count"] = edges_gdf["accident_count"].fillna(0).astype(int)
    # 10) 写回图：把 accident_count 写到每条边
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        data["accident_count"] = 0  # 先重置为0，避免残留

    for row in edges_gdf.itertuples(index=False):
        # 依赖你 build_edge_gdf 里带的 u, v, key 三列
        G_proj.edges[row.u, row.v, row.key]["accident_count"] = int(row.accident_count)

    # 11) 保存 CSV（仅常用字段）
    edges_gdf[["u", "v", "key", "length_m", "accident_count"]].to_csv(
        OUT_EDGES_CSV, index=False, encoding="utf-8-sig"
    )

    # 12) 保存 GraphML（后续路径规划直接加载这个）
    ox.save_graphml(G_proj, filepath=OUT_GRAPH_GRAPHML)

    print(f"✅ 已保存：{OUT_EDGES_CSV}")
    print(f"✅ 已保存：{OUT_GRAPH_GRAPHML}")

if __name__ == "__main__":
    main()
