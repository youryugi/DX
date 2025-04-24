import os
import pickle
import time
from datetime import datetime, timezone, timedelta

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.ops import unary_union

# 如果你需要投影或用到Transformer:
from pyproj import Transformer

def precompute_shadow_ratios():
    """
    批量计算 (边, 时间片) 的阴影覆盖比例 shadow_ratio = 阴影长度 / 边总长度
    并把结果保存到 pkl 文件中，方便主程序直接读取，而不再做实时 intersection。
    """

    # ---------------------（1）加载建筑 & 阴影数据---------------------
    print("开始加载建筑和阴影数据...")
    # 这里演示读取你给定的建筑大合集 bldg_merged_gdf.pkl
    bldg_pkl = r"bldg_merged_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl"
    with open(bldg_pkl,'rb') as f:
        building_gdf = pickle.load(f)

    # 确保投影为EPSG:6669
    if building_gdf.crs.to_epsg() != 6669:
        building_gdf = building_gdf.to_crs(epsg=6669)

    # 加载阴影 {time: geometry} 字典
    # 例如 5分钟间隔的那个 .pkl
    shadow_file = r"shadows_20241205_1300_1400_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl"
    with open(shadow_file, 'rb') as f:
        time_to_union = pickle.load(f)
    # time_to_union: dict[datetime -> MultiPolygon/Polygon/... or None]

    # ---------------------（2）加载或获取 OSM 图 & 转成EPSG:6669---------------------
    print("开始加载OSM图...")
    # 先拼一下 bbox
    building_gdf_wgs84 = building_gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = building_gdf_wgs84.total_bounds
    # OSMnx 需要 (北, 南, 东, 西)
    north, south, east, west = maxy, miny, maxx, minx

    # 你已有一个本地缓存 pkl，就从里面读。如果没有则下载。
    map_id = f"{north}_{south}_{east}_{west}"
    osm_file = f"osmnx_graph_{map_id}.pkl"
    if os.path.exists(osm_file):
        with open(osm_file,"rb") as f:
            G = pickle.load(f)
    else:
        G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type="bike")
        with open(osm_file,"wb") as f:
            pickle.dump(G,f)

    # 提取边
    gdf_edges = ox.graph_to_gdfs(G, nodes=False)
    # 如果原始是 4326，则转成 6669
    if gdf_edges.crs.to_epsg() != 4326:
        gdf_edges = gdf_edges.to_crs(epsg=4326)
    gdf_edges = gdf_edges.to_crs(epsg=6669)

    # ---------------------（3）准备批量 intersection---------------------
    print("开始批量计算每条边在每个时间片的阴影覆盖比...")

    # 我们把结果做成一个 dict 或 DataFrame
    # 这里先演示用 dict: precomputed[(u,v,k, t)] = ratio
    precomputed = {}

    # 对时间排序，保证处理顺序
    sorted_times = sorted(time_to_union.keys())

    total_edges = len(gdf_edges)
    start_time_all = time.time()
    count_inter = 0

    for i, (u,v,k) in enumerate(gdf_edges.index):
        edge_geom = gdf_edges.loc[(u,v,k),'geometry']
        edge_len = edge_geom.length

        # 处理每个时间片
        for t in sorted_times:
            shadow_poly = time_to_union[t]
            if shadow_poly is None or shadow_poly.is_empty:
                ratio = 0.0
            else:
                # 做 intersection
                count_inter += 1
                inters = edge_geom.intersection(shadow_poly)
                shadow_len = inters.length if (not inters.is_empty) else 0.0
                ratio = shadow_len / edge_len if edge_len > 0 else 0.0

            precomputed[(u,v,k,t)] = ratio

        if (i+1) % 100 == 0:
            print(f"已处理 {i+1}/{total_edges} 条边...")

    end_time_all = time.time()
    print("==========离线批量计算完成==========")
    print(f"共处理 {total_edges} 条边 × {len(sorted_times)} 个时间 -> {count_inter} 次 intersection()")
    print(f"总耗时约: {end_time_all - start_time_all:.2f} 秒")

    # ---------------------（4）保存到 pkl---------------------
    # 从 shadow_file 提取文件名中的时间与经纬度信息（假设命名规范一致）
    shadow_filename = os.path.basename(shadow_file).replace("shadows_", "").replace(".pkl", "")
    # 拼接输出文件名
    out_file = f"edge_shadow_ratios_{shadow_filename}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(precomputed, f)

    print(f"已将结果写入 {out_file}，可以在主程序中直接读取使用！")


if __name__ == '__main__':
    precompute_shadow_ratios()
