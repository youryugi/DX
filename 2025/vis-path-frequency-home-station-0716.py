# -*- coding: utf-8 -*-
"""
Shortest‑path edge‑frequency analysis between residential buildings (usage 411/412)
and the nearest railway station ("駅") in a given area.

Key fixes compared with the previous draft
-----------------------------------------
* Added **default_color** and **usage_color_map** so that every building gets a colour.
* Replaced all phantom references to usage "422" with the explicit keyword "station".
* Unified variable names:  bldg_station   (instead of pseudo‑"422")
* Guarded against missing CRS information; converts to EPSG:4326 before graph download.
* Collected configurable parameters (padding, projected CRS, 412 frequency multiplier) at the top.
* Added simple progress messages and exception handling for robustness.
"""

import os
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from shapely.ops import nearest_points  # noqa: F401  (kept for future use)
from matplotlib.patches import Patch

# --------------------------------------------------
# 1) Configurable parameters
# --------------------------------------------------
PADDING_DEG = 0.001        # ~110 m latitude‑wise
PROJECTED_CRS = 6669       # JGD2011 / Japan Plane Rectangular CS II (fits Kobe‑Osaka‑Kyoto)
FREQ_MULT_412 = 22.5       # weight for apartment trips

# Colours
DEFAULT_COLOR = "#d9d9d9"  # light grey for unclassified buildings
USAGE_COLOR_MAP = {
    "411": "#4daf4a",   # green   – Single‑family house
    "412": "#377eb8",   # blue    – Apartment / Condominium
    "station": "#ff7f00"  # orange  – 駅  (railway station)
}

# --------------------------------------------------
# 2) Read building footprints (CityGML / PLATEAU)
# --------------------------------------------------
print("1. 正在读取建筑数据…")
BLDG_GML_FILES = [
    r"bldg/51357451_bldg_6697_op.gml",
    # r"bldg/51357452_bldg_6697_op.gml",
    # r"bldg/51357453_bldg_6697_op.gml",
    # r"bldg/51357461_bldg_6697_op.gml",
    # r"bldg/51357462_bldg_6697_op.gml",
    # r"bldg/51357463_bldg_6697_op.gml",
]

building_gdf = gpd.GeoDataFrame(
    pd.concat([gpd.read_file(f) for f in BLDG_GML_FILES], ignore_index=True)
)

# Filter: usage starts with 411/412 OR name contains "駅"
print("   按 usage 与名称过滤…")
is_411 = building_gdf["usage"].astype(str).str.startswith("411")
is_412 = building_gdf["usage"].astype(str).str.startswith("412")
is_station = building_gdf["name"].fillna("").str.contains("駅", regex=False)

building_gdf = building_gdf[is_411 | is_412 | is_station].copy()

# Assign colours
building_gdf["color"] = DEFAULT_COLOR
building_gdf.loc[is_411, "color"] = USAGE_COLOR_MAP["411"]
building_gdf.loc[is_412, "color"] = USAGE_COLOR_MAP["412"]
building_gdf.loc[is_station, "color"] = USAGE_COLOR_MAP["station"]

print("   过滤后建筑总数:", len(building_gdf))
print("   usage 唯一值:", building_gdf["usage"].unique())

# --------------------------------------------------
# 3) Download / cache drive network inside bounding box
# --------------------------------------------------
print("2. 计算建筑物范围并下载对应路网…")
if building_gdf.crs is None:
    # Assume source CRS is already EPSG:4326 (most PLATEAU GML files are)
    building_gdf.set_crs(epsg=4326, inplace=True)
elif building_gdf.crs.to_epsg() != 4326:
    building_gdf = building_gdf.to_crs(epsg=4326)

minx, miny, maxx, maxy = building_gdf.total_bounds
minx -= PADDING_DEG
miny -= PADDING_DEG
maxx += PADDING_DEG
maxy += PADDING_DEG
print(f"   路网下载范围: ({miny:.5f}, {maxy:.5f}, {minx:.5f}, {maxx:.5f})")

bbox_str = f"{miny:.5f}_{maxy:.5f}_{minx:.5f}_{maxx:.5f}"
GRAPHML_PATH = f"cached_network_{bbox_str}.graphml"

if os.path.exists(GRAPHML_PATH):
    print(f"   已检测到缓存文件 {GRAPHML_PATH}，直接读取…")
    G = ox.load_graphml(GRAPHML_PATH)
else:
    print("   无缓存，开始下载路网 (network_type='drive')…")
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type="drive")
    ox.save_graphml(G, GRAPHML_PATH)
    print(f"   路网已保存到 {GRAPHML_PATH}")

print("   路网加载完成: 节点", len(G.nodes), "边", len(G.edges))

# --------------------------------------------------
# 4) Prepare centroid lists (projected CRS for distance calcs)
# --------------------------------------------------
print("3. 计算住宅 / 駅 的质心…")
if building_gdf.crs.to_epsg() != PROJECTED_CRS:
    building_gdf_proj = building_gdf.to_crs(epsg=PROJECTED_CRS)
else:
    building_gdf_proj = building_gdf

bldg_411 = building_gdf_proj[building_gdf_proj["usage"].astype(str).str.startswith("411")]
bldg_412 = building_gdf_proj[building_gdf_proj["usage"].astype(str).str.startswith("412")]
bldg_station = building_gdf_proj[is_station]

if bldg_411.empty or bldg_station.empty:
    raise RuntimeError("未找到 411 或 駅 建筑，无法继续。")

bldg_411_cent = bldg_411.geometry.centroid
bldg_412_cent = bldg_412.geometry.centroid
bldg_station_cent = bldg_station.geometry.centroid

# --------------------------------------------------
# 5) Helper to find nearest station and accumulate edge frequency
# --------------------------------------------------
print("4. 正在计算最短路径并统计边频率…")
edge_counter: Counter = Counter()
all_paths = []

# Project road graph once for plotting
G_proj = ox.project_graph(G, to_crs=f"EPSG:{PROJECTED_CRS}")

def accumulate_paths(orig_series, multiplier=1.0):
    for i, orig_pt in enumerate(orig_series):
        # 最近 station
        dists = bldg_station_cent.distance(orig_pt)
        nearest_idx = dists.idxmin()
        dest_pt = bldg_station_cent.loc[nearest_idx]

        # 质心 -> WGS84 for OSMnx NN lookup
        orig_wgs = gpd.GeoSeries([orig_pt], crs=PROJECTED_CRS).to_crs(epsg=4326).iloc[0]
        dest_wgs = gpd.GeoSeries([dest_pt], crs=PROJECTED_CRS).to_crs(epsg=4326).iloc[0]
        try:
            orig_node = ox.distance.nearest_nodes(G, X=orig_wgs.x, Y=orig_wgs.y)
            dest_node = ox.distance.nearest_nodes(G, X=dest_wgs.x, Y=dest_wgs.y)
            path = nx.shortest_path(G, orig_node, dest_node, weight="length")
            all_paths.append(path)
            for u, v in zip(path[:-1], path[1:]):
                # Use consistent (u,v) ordering
                edge = (u, v) if G.has_edge(u, v) else (v, u)
                edge_counter[edge] += multiplier
        except nx.NetworkXNoPath:
            print(f"   * 无路可达: 原点{i+1}")
            continue
        if (i + 1) % 100 == 0 or (i + 1) == len(orig_series):
            print(f"   已处理 {i+1}/{len(orig_series)} 条记录 (mult={multiplier})")

# 5a) 411 → 駅  (权重 1)
accumulate_paths(bldg_411_cent, multiplier=1.0)

# 5b) 412 → 駅  (权重 FREQ_MULT_412)
if not bldg_412_cent.empty:
    accumulate_paths(bldg_412_cent, multiplier=FREQ_MULT_412)

# --------------------------------------------------
# 6) Visualisation
# --------------------------------------------------
print("5. 绘图…")
fig, ax = plt.subplots(figsize=(12, 8))
ox.plot_graph(
    G_proj,
    ax=ax,
    show=False,
    close=False,
    edge_color="lightgray",
    edge_linewidth=0.5,
    node_size=0,
)

# Building footprints
building_gdf_proj.plot(
    ax=ax,
    color=building_gdf_proj["color"],
    edgecolor=None,
    linewidth=0.3,
    alpha=0.9,
)

# Edge frequency lines
if edge_counter:
    max_freq = max(edge_counter.values())
    for (u, v), freq in edge_counter.items():
        data = G_proj.get_edge_data(u, v)
        geom = data[0].get("geometry") if data else None
        if geom is None:
            geom_x = [G_proj.nodes[u]["x"], G_proj.nodes[v]["x"]]
            geom_y = [G_proj.nodes[u]["y"], G_proj.nodes[v]["y"]]
        else:
            geom_x, geom_y = geom.xy
        ax.plot(
            geom_x,
            geom_y,
            color="red",
            linewidth=1 + 4 * freq / max_freq,
            alpha=0.8,
        )

# Legend
legend_handles = [
    Patch(facecolor=USAGE_COLOR_MAP["411"], label="Single‑family "),
    Patch(facecolor=USAGE_COLOR_MAP["412"], label="Apartment "),
    Patch(facecolor=USAGE_COLOR_MAP["station"], label="Station"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=12)
ax.set_title("Edge frequency: Residential → Station", fontsize=16)
out_tag = bbox_str.replace("_", "-")  # shorter
plt.tight_layout()
plt.savefig(f"edge_freq_home-station_{out_tag}.png", dpi=300)
plt.show()

# --------------------------------------------------
# 7) Save edge frequency as GPKG + CSV
# --------------------------------------------------
print("6. 保存边频率数据…")
records = []
for (u, v), freq in edge_counter.items():
    data = G_proj.get_edge_data(u, v)
    geom = data[0].get("geometry") if data else None
    records.append({"u": u, "v": v, "freq": freq, "geometry": geom})

freq_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=f"EPSG:{PROJECTED_CRS}")

out_tag = bbox_str.replace("_", "-")  # shorter
freq_gdf.to_file(f"edge_freq_home-station_{out_tag}.gpkg", layer="edge_freq", driver="GPKG")
freq_gdf.drop(columns="geometry").to_csv(f"edge_freq_home-station_{out_tag}.csv", index=False)
print("   完成！")