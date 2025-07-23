# -*- coding: utf-8 -*-
"""
Shortest‑path edge‑frequency analysis between school buildings (usage 422)
and the nearest railway station ("駅") in a given area.

Key updates for the 422‑only version
------------------------------------
* Switched every reference from 411/412 (residential) to 422 (schools).
* Removed the apartment‑specific frequency multiplier logic.
* Added a dedicated colour for schools in the map legend.
* Simplified the path‑accumulation section (single origin category).
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
import time  # 新增

start_time = time.time()  # 记录开始时间
# --------------------------------------------------
# 1) Configurable parameters
# --------------------------------------------------
PADDING_DEG = 0.001        # ~110 m latitude‑wise
PROJECTED_CRS = 6669       # JGD2011 / Japan Plane Rectangular CS II (fits Kobe‑Osaka‑Kyoto)

# Colours
DEFAULT_COLOR = "#d9d9d9"  # light grey for unclassified buildings
USAGE_COLOR_MAP = {
    "422": "#984ea3",   # purple – School
    "station": "#ff7f00"  # orange – 駅  (railway station)
}

# --------------------------------------------------
# 2) Read building footprints (CityGML / PLATEAU)
# --------------------------------------------------
print("1. 正在读取建筑数据…")
BLDG_GML_FILES = [
        r"../time-shadow/bldg/51357399_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357490_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357491_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357492_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357493_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357389_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357480_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357481_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357482_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357483_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357379_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357470_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357471_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357472_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357473_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357369_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357460_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357461_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357462_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357463_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357359_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357450_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357451_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357452_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357453_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357349_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357440_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357441_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357442_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357443_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357339_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357430_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357431_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357432_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357433_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357329_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357420_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357421_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357422_bldg_6697_op.gml",
        r"../time-shadow/bldg/51357423_bldg_6697_op.gml",

        #r"bldg\51357462_bldg_6697_op.gml",
        #r"bldg\51357463_bldg_6697_op.gml",
    # r"bldg\51357451_bldg_6697_op.gml",
    # r"bldg\51357452_bldg_6697_op.gml",
    # r"bldg\51357453_bldg_6697_op.gml",
    # r"bldg\51357461_bldg_6697_op.gml",
    # r"bldg\51357462_bldg_6697_op.gml",
    # r"bldg\51357463_bldg_6697_op.gml",
    # r"bldg\51357471_bldg_6697_op.gml",
    # r"bldg\51357472_bldg_6697_op.gml",
    # r"bldg\51357473_bldg_6697_op.gml"
]

building_gdf = gpd.GeoDataFrame(
    pd.concat([gpd.read_file(f) for f in BLDG_GML_FILES], ignore_index=True)
)

# Filter: usage == 422 OR name contains "駅"
print("   按 usage 与名称过滤…")
is_422 = building_gdf["usage"].astype(str).str.startswith("422")
is_station = building_gdf["name"].fillna("").str.contains("駅", regex=False)

building_gdf = building_gdf[is_422 | is_station].copy()

# Assign colours
building_gdf["color"] = DEFAULT_COLOR
building_gdf.loc[is_422, "color"] = USAGE_COLOR_MAP["422"]
building_gdf.loc[is_station, "color"] = USAGE_COLOR_MAP["station"]

print("   过滤后建筑总数:", len(building_gdf))
print("   usage 唯一值:", building_gdf["usage"].unique())

# --------------------------------------------------
# 3) Download / cache drive network inside bounding box
# --------------------------------------------------
print("2. 计算建筑物范围并下载对应路网…")
if building_gdf.crs is None:
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
print("3. 计算学校 / 駅 的质心…")
if building_gdf.crs.to_epsg() != PROJECTED_CRS:
    building_gdf_proj = building_gdf.to_crs(epsg=PROJECTED_CRS)
else:
    building_gdf_proj = building_gdf

bldg_422 = building_gdf_proj[building_gdf_proj["usage"].astype(str).str.startswith("422")]
bldg_station = building_gdf_proj[is_station]

if bldg_422.empty or bldg_station.empty:
    raise RuntimeError("未找到 422 学校或 駅 建筑，无法继续。")

bldg_422_cent = bldg_422.geometry.centroid
bldg_station_cent = bldg_station.geometry.centroid

# --------------------------------------------------
# 5) Helper to find nearest station and accumulate edge frequency
# --------------------------------------------------
print("4. 正在计算最短路径并统计边频率…")
edge_counter: Counter = Counter()
all_paths = []

# Project road graph once for plotting
G_proj = ox.project_graph(G, to_crs=f"EPSG:{PROJECTED_CRS}")

def accumulate_paths(orig_series):
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
                edge_counter[edge] += 1.0
        except nx.NetworkXNoPath:
            print(f"   * 无路可达: 原点{i+1}")
            continue
        if (i + 1) % 100 == 0 or (i + 1) == len(orig_series):
            print(f"   已处理 {i+1}/{len(orig_series)} 条记录")

# 5a) 422 → 駅
accumulate_paths(bldg_422_cent)

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
    Patch(facecolor=USAGE_COLOR_MAP["422"], label="School"),
    Patch(facecolor=USAGE_COLOR_MAP["station"], label="Station"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=12)
ax.set_title("Edge frequency: School → Station", fontsize=16)
out_tag = bbox_str.replace("_", "-")  # shorter
plt.tight_layout()
plt.savefig(f"edge_freq_school-station_{out_tag}.png", dpi=300)
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

out_tag = bbox_str.replace("_", "-")
freq_gdf.to_file(f"edge_freq_school-station_{out_tag}.gpkg", layer="edge_freq", driver="GPKG")
freq_gdf.drop(columns="geometry").to_csv(f"edge_freq_school-station_{out_tag}.csv", index=False)
print("   完成！")
# 输出总用时
elapsed = time.time() - start_time
print(f"总用时: {elapsed:.1f} 秒")