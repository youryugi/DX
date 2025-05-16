import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astral import LocationInfo
from astral.sun import elevation, azimuth
from datetime import datetime, timezone, timedelta
import osmnx as ox
import networkx as nx
from shapely.ops import unary_union
from shapely.affinity import translate
from matplotlib.widgets import Slider, Button
from pyproj import Transformer
from matplotlib.lines import Line2D
import pandas as pd
import geopandas as gpd
import os
import time
import pickle
import os, time, math, pickle, heapq
# ---------------------------------------------------------------------------------------
# 1) 读取离线计算好的 (u,v,k, time) -> shadow_ratio
# 注意：请确认与下面的 G、时间切分一致
# --------更新cost-------------------------------------------------------------------------------

# 路径设置
bldg_gml_files = [
    r"bldg\51357451_bldg_6697_op.gml",
    r"bldg\51357452_bldg_6697_op.gml",
    r"bldg\51357453_bldg_6697_op.gml",
    r"bldg\51357461_bldg_6697_op.gml",
    r"bldg\51357462_bldg_6697_op.gml",
    r"bldg\51357463_bldg_6697_op.gml",
    r"bldg\51357471_bldg_6697_op.gml",
    r"bldg\51357472_bldg_6697_op.gml",
    r"bldg\51357473_bldg_6697_op.gml"
]
road_gml_files = [
    r"tran\51357451_tran_6697_op.gml",
    r"tran\51357452_tran_6697_op.gml",
    r"tran\51357453_tran_6697_op.gml",
    r"tran\51357461_tran_6697_op.gml",
    r"tran\51357462_tran_6697_op.gml",
    r"tran\51357463_tran_6697_op.gml",
    r"tran\51357471_tran_6697_op.gml",
    r"tran\51357472_tran_6697_op.gml",
    r"tran\51357473_tran_6697_op.gml"
]
# pkl保存路径
pkl_path = r"bldg_merged_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl"
#先加载边的ratio
with open("edge_shadow_ratios_20241205_0900_1000_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl", "rb") as f:
    precomputed = pickle.load(f)
shadow_file = r"shadows_20241205_0900_1000_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl"
#设置固定点还是手动
manual_input_mode = True
manual_origin_point_wgs84 = (34.632734242239636, 135.51493454131852)
manual_destination_point_wgs84 = (34.63049080065106, 135.54547444776205)
manual_origin_point_wgs84 = (34.62709838787363, 135.5151808481631)#大地图
manual_destination_point_wgs84 = (34.64854640692838, 135.54677174184593)#大地图
# ---------------------------------------------------------------------------------------
# 统计 intersection 次数（如果你想去掉，可以删除）
# ---------------------------------------------------------------------------------------
intersection_counter = 0

# 在代码开头处定义图例句柄
shortest_route_legend = Line2D([0], [0], color='red', linewidth=2, label='Shortest Bike Route')
wanted_route_legend   = Line2D([0], [0], color='green', linewidth=2, label='Wanted Bike Route (Dynamic A*)')
static_route_legend = Line2D([0], [0], color='blue', linewidth=2, label='Static A* Route')
bigfontsize = 14



# 计时开始
start_time = time.time()

if os.path.exists(pkl_path):
    print("发现已有.pkl缓存，直接加载...")
    with open(pkl_path, 'rb') as f:
        bldg_merged_gdf = pickle.load(f)
else:
    print("未发现.pkl缓存，开始读取GML并合并...")
    bldg_gdf_list = [gpd.read_file(file) for file in bldg_gml_files]
    bldg_merged_gdf = pd.concat(bldg_gdf_list, ignore_index=True)

    print("保存为.pkl...")
    with open(pkl_path, 'wb') as f:
        pickle.dump(bldg_merged_gdf, f)

end_time = time.time()
print("读取和处理耗时:", end_time - start_time, "秒")
print(bldg_merged_gdf.head())

# ---------------------------------------------------------------------------------------
# 读取 tran GML
# ---------------------------------------------------------------------------------------
starttime2 = time.time()

road_gdf_list = [gpd.read_file(file) for file in road_gml_files]
merged_road_gdf = pd.concat(road_gdf_list, ignore_index=True)
endtime2 = time.time()
readtrantime = endtime2 - starttime2
print("读取tran的时间", readtrantime)

building_gdf = bldg_merged_gdf
road_gdf = merged_road_gdf

# 确保投影为EPSG:6669
if building_gdf.crs.to_epsg() != 6669:
    building_gdf = building_gdf.to_crs(epsg=6669)
if road_gdf.crs.to_epsg() != 6669:
    road_gdf = road_gdf.to_crs(epsg=6669)

# ---------------------------------------------------------------------------------------
# calculate_shadow_stats (若只做演示，可留着；否则可改为也使用 precomputed)
# ---------------------------------------------------------------------------------------
def calculate_shadow_stats(route_gdf, time_to_union, start_time, coef, sample_interval=60):
    speed = 10 * 1000 / 3600  # 10km/h -> m/s
    global intersection_counter
    current_time = start_time

    shadow_distance = 0.0
    non_shadow_distance = 0.0
    total_distance = 0.0

    for idx, row in route_gdf.iterrows():
        edge_geom = row.geometry
        edge_length = edge_geom.length
        travel_time_s = edge_length / speed

        temp_time = current_time
        remaining_time = travel_time_s

        while remaining_time > 0:
            dt = min(sample_interval, remaining_time)
            mid_time = temp_time + timedelta(seconds=dt / 2)

            nearest_time = find_nearest_time(time_to_union.keys(), mid_time)
            shadow_union = time_to_union[nearest_time]

            if shadow_union:
                intersection_counter += 1
                # 原先的 intersection，可留作参考
                inters = edge_geom.intersection(shadow_union)
                shadow_len = inters.length if not inters.is_empty else 0
            else:
                shadow_len = 0

            ratio = dt / travel_time_s
            shadow_ratio = shadow_len / edge_length if edge_length > 0 else 0

            shadow_distance += edge_length * shadow_ratio * ratio
            non_shadow_distance += edge_length * (1 - shadow_ratio) * ratio

            temp_time += timedelta(seconds=dt)
            remaining_time -= dt

        current_time += timedelta(seconds=travel_time_s)
        total_distance += edge_length

    print(f"阴影下路程: {shadow_distance:.2f} m, "
          f"阳光下路程: {non_shadow_distance:.2f} m, "
          f"总路程: {total_distance:.2f} m")
    print("计算了", intersection_counter)

# ---------------------------------------------------------------------------------------
# 读取 time_to_union（每个时刻的阴影区域），并找 nearest_time
# ---------------------------------------------------------------------------------------

with open(shadow_file, 'rb') as f:
    time_to_union = pickle.load(f)

def find_nearest_time(keys, target):
    return min(keys, key=lambda t: abs(t - target))

date_time = datetime(2024, 12, 5, 9, 10, tzinfo=timezone(timedelta(hours=9)))
nearest_time = find_nearest_time(time_to_union.keys(), date_time)
shadow_union = time_to_union[nearest_time]

if shadow_union is None:
    print("该时刻太阳在地平线以下，或无有效阴影。")
    exit()

shadow_gdf = gpd.GeoDataFrame(geometry=[shadow_union], crs='EPSG:6669')

# ---------------------------------------------------------------------------------------
# 准备绘图
# ---------------------------------------------------------------------------------------
plt.rcParams['font.family'] = 'SimHei'
fig, ax = plt.subplots(figsize=(12, 8))

bounds = building_gdf.total_bounds
buffer = 10
x_min, y_min, x_max, y_max = bounds
x_min -= buffer
y_min -= buffer
x_max += buffer
y_max += buffer

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

road_gdf.plot(ax=ax, color='yellow', alpha=0.5, label='road')
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='shadow')
building_gdf.plot(ax=ax, color='lightblue', label='building')

legend_handles = [
    Patch(facecolor='yellow',    label='Road'),
    Patch(facecolor='lightblue', label='Building'),
    Patch(facecolor='gray', edgecolor='gray', label='Shadow'),
    Line2D([0], [0], marker='o', color='blue', markersize=8,  linestyle='None', label='Start Point'),
    Line2D([0], [0], marker='o', color='magenta', markersize=8, linestyle='None', label='End Point'),
    Line2D([0], [0], color='red',   linewidth=2, label='Shortest Route'),
    Line2D([0], [0], color='green', linewidth=2, label='Wanted Route \n (Dynamic shadow)'),
    Line2D([0], [0], color='orange', linewidth=2, label='Wanted Route \n (Static shadow)'),
]
plt.title("Roads, buildings, and shadows     ↑North", fontsize=16)
plt.legend(handles=legend_handles, loc='lower right', fontsize=bigfontsize)
plt.xlabel("X (m)", fontsize=bigfontsize)
plt.ylabel("Y (m)", fontsize=bigfontsize)
plt.xticks(fontsize=bigfontsize)
plt.yticks(fontsize=bigfontsize)
plt.grid(True)

# ---------------------------------------------------------------------------------------
# 获取 OSMnx 图
# ---------------------------------------------------------------------------------------
building_gdf_wgs84 = building_gdf.to_crs(epsg=4326)
building_bounds_wgs84 = building_gdf_wgs84.total_bounds
bbox = (building_bounds_wgs84[3], building_bounds_wgs84[1],
        building_bounds_wgs84[2], building_bounds_wgs84[0])
map_id = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
osm_file = f"osmnx_graph_{map_id}.pkl"

if os.path.exists(osm_file):
    print(f"加载本地OSM数据: {osm_file}")
    with open(osm_file, "rb") as f:
        G = pickle.load(f)
else:
    print("开始下载OSM数据...")
    G = ox.graph_from_bbox(north=bbox[0], south=bbox[1], east=bbox[2], west=bbox[3], network_type="bike")
    print('下载完成')
    with open(osm_file, "wb") as f:
        pickle.dump(G, f)

print(f"节点数量: {len(G.nodes)}")
print(f"边数量: {len(G.edges)}")

# 提取边为 gdf_edges 并投影EPSG:6669
gdf_edges = ox.graph_to_gdfs(G, nodes=False)
if gdf_edges.crs.to_epsg() != 4326:
    gdf_edges = gdf_edges.to_crs(epsg=4326)
gdf_edges = gdf_edges.to_crs(epsg=6669)

# 合并阴影(仅用于可视化)
shadow_union = unary_union(shadow_gdf.geometry)

# ---------------------------------------------------------------------------------------
# 交互选点
# ---------------------------------------------------------------------------------------
transformer_to_wgs84 = Transformer.from_crs(6669, 4326, always_xy=True)
click_count = 0
origin_point_wgs84 = None
destination_point_wgs84 = None
origin_marker = None
destination_marker = None
#

proj_crs = 'EPSG:6669'
transformer_from_wgs84 = Transformer.from_crs('EPSG:4326', proj_crs, always_xy=True)

def on_map_click(event):
    global click_count, origin_point_wgs84, destination_point_wgs84
    global origin_marker, destination_marker

    if manual_input_mode:
        if click_count == 0:
            origin_point_wgs84 = manual_origin_point_wgs84
            x, y = transformer_from_wgs84.transform(origin_point_wgs84[1], origin_point_wgs84[0])
            origin_marker = ax.plot(x, y, marker='o', color='blue', markersize=8, label='Origin')[0]
            plt.draw()
            click_count += 1
            print(f"复现起点: (lat={origin_point_wgs84[0]}, lon={origin_point_wgs84[1]})")
        elif click_count == 1:
            destination_point_wgs84 = manual_destination_point_wgs84
            x, y = transformer_from_wgs84.transform(destination_point_wgs84[1], destination_point_wgs84[0])
            destination_marker = ax.plot(x, y, marker='o', color='magenta', markersize=8, label='Destination')[0]
            plt.draw()
            click_count += 1
            print(f"复现终点: (lat={destination_point_wgs84[0]}, lon={destination_point_wgs84[1]})")
        return

    if event.inaxes != ax:
        return

    x_coord, y_coord = event.xdata, event.ydata
    lon, lat = transformer_to_wgs84.transform(x_coord, y_coord)

    if click_count == 0:
        origin_point_wgs84 = (lat, lon)
        if origin_marker is not None:
            origin_marker.remove()
        origin_marker = ax.plot(x_coord, y_coord, marker='o', color='blue', markersize=8, label='Origin')[0]
        plt.draw()
        click_count += 1
        print(f"已选择起点: (lat={lat}, lon={lon})")
    elif click_count == 1:
        destination_point_wgs84 = (lat, lon)
        if destination_marker is not None:
            destination_marker.remove()
        destination_marker = ax.plot(x_coord, y_coord, marker='o', color='magenta', markersize=8, label='Destination')[0]
        plt.draw()
        click_count += 1
        print(f"已选择终点: (lat={lat}, lon={lon})")

fig.canvas.mpl_connect('button_press_event', on_map_click)

start_time = date_time

# ---------------------------------------------------------------------------------------
# 重点：在多状态 Dijkstra 里使用 precomputed，而不再 intersection()
# ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 1. 构建边阴影趋势查询表（IntervalIndex）并计时
# ----------------------------------------------------------------------------

def build_trend_interval_map(edge_trend_map):
    """将 {edge: [ {start,end,type}, ... ]} -> {edge: (IntervalIndex, [type])}"""
    import pandas as pd
    t0 = time.time()
    tmap = {}
    for key, segs in edge_trend_map.items():
        starts = [s['start'] for s in segs]
        ends   = [s['end']   for s in segs]
        types  = [s['type']  for s in segs]
        tmap[key] = (pd.IntervalIndex.from_arrays(starts, ends, closed='left'), types)
    print(f"[计时] trend_interval_map 构建完毕: {time.time()-t0:.4f}s  (edges={len(tmap)})")
    return tmap

# 由外部脚本提供 edge_trend_map
auto_tmap_start = time.time()
with open("edge_trend_map_20241205_0900_1000_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl", "rb") as f:
    edge_trend_map = pickle.load(f)
trend_interval_map = build_trend_interval_map(edge_trend_map)
print(f"trend_interval_map 构建总耗时记录: {(time.time()-auto_tmap_start):.4f}s\n")

# ----------------------------------------------------------------------------
# 2. A* 搜索：多状态 + 时变代价 + 全流程计时
# ----------------------------------------------------------------------------

def update_cool_route(coef, start_time, sample_interval=300):
    """返回 GeoDataFrame: 最优阴影‑感知路径  + 控制台打印完整计时统计"""

    # --- 起终点 ---
    origin_node = ox.distance.nearest_nodes(G, X=manual_origin_point_wgs84[1],      Y=manual_origin_point_wgs84[0])
    goal_node   = ox.distance.nearest_nodes(G, X=manual_destination_point_wgs84[1], Y=manual_destination_point_wgs84[0])

    # =============== 全局计时器 ===============
    t_global_start       = time.time()
    t_cost_accum         = 0.0   # time‑dependent cost 累计
    t_heuristic_accum    = 0.0   # heuristic 累计
    t_neighbor_accum     = 0.0   # 邻居遍历累积
    td_calls             = 0     # cost 调用计数
    h_calls              = 0     # heuristic 调用计数
    neighbor_visits      = 0     # 邻居计数

    # =============== 函数定义 ===============
    speed = 10 * 1000 / 3600  # m/s
    goal_lat, goal_lon = G.nodes[goal_node]['y'], G.nodes[goal_node]['x']

    def heuristic(node, cur_time_s):
        nonlocal t_heuristic_accum, h_calls
        t0 = time.time(); h_calls += 1
        lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
        dist = ox.distance.euclidean_dist_vec(lat, lon, goal_lat, goal_lon)
        # 估计抵达该直线距离所需时间 (保守取直线)
        est_dt = cur_time_s + dist / speed
        future_dt = start_time + timedelta(seconds=est_dt)
        minute = future_dt.hour * 60 + future_dt.minute
        ratio = 0.5  # 保守设全阳，可替换为更复杂预测
        h_val = dist * (1 + coef * ratio)
        t_heuristic_accum += (time.time() - t0)
        return h_val

    def time_dependent_cost(u, v, k, arrival_dt):
        """按 sample_interval 对边进行积分，返回 (cost, duration)"""
        nonlocal t_cost_accum, td_calls
        t0 = time.time(); td_calls += 1

        if (u, v, k) not in gdf_edges.index:
            return math.inf, math.inf
        geom = gdf_edges.loc[(u, v, k), 'geometry']
        length = geom.length
        duration = length / speed
        cost_acc = 0.0
        tmp_dt   = arrival_dt
        remain   = duration
        while remain > 0:
            dt = min(sample_interval, remain)
            mid = tmp_dt + timedelta(seconds=dt/2)
            nt  = find_nearest_time(time_to_union.keys(), mid)
            ratio = precomputed.get((u, v, k, nt), 0.0) if nt else 0.0
            sunny_len = length * (1 - ratio)
            #cost_acc += (sunny_len + coef * length) * (dt / duration)老的cost了 也需要更新
            #step_cost = length - coef * shadow_ratio * length
            #cost = sunny_dist + coef * edge_length 这个好像才对？
            cost_acc +=sunny_len+ coef * length
            tmp_dt  += timedelta(seconds=dt)
            remain  -= dt
        t_cost_accum += (time.time() - t0)
        return cost_acc, duration

    # =============== A* 主循环 ===============
    open_q = []
    heapq.heappush(open_q, (heuristic(origin_node, 0.0), origin_node, 0.0))
    g_best = {origin_node: 0.0}
    pred   = {}              # (node,time) -> (prev_node, prev_time)

    while open_q:
        f_val, cur, cur_t = heapq.heappop(open_q)
        if cur == goal_node:
            break
        arr_dt = start_time + timedelta(seconds=cur_t)

        t_nb0 = time.time()
        for nb, edict in G[cur].items():
            for k in edict:
                inc_cost, dur = time_dependent_cost(cur, nb, k, arr_dt)
                if math.isinf(inc_cost):
                    continue
                new_t = cur_t + dur
                new_g = g_best[cur] + inc_cost
                if nb not in g_best or new_g < g_best[nb] - 1e-6:
                    g_best[nb] = new_g
                    pred[(nb, new_t)] = (cur, cur_t)
                    heapq.heappush(open_q, (new_g + heuristic(nb, new_t), nb, new_t))
                    neighbor_visits += 1
        t_neighbor_accum += (time.time() - t_nb0)

    if goal_node not in g_best:
        print("[A*] 未找到可行路径\n"); return None

    # =============== 回溯最晚时间戳状态 ===============
    goal_states = [(t, key) for key,t in [(k[0],k[1]) for k in pred if k[0]==goal_node]]
    best_t = max([t for t,_ in goal_states])
    nodes = [goal_node]; cur = (goal_node, best_t)
    while cur in pred:
        prev = pred[cur]; nodes.append(prev[0]); cur = prev
    nodes.reverse()

    # =============== 组装 GeoDataFrame ===============
    edges=[]
    for i in range(len(nodes)-1):
        u,v = nodes[i], nodes[i+1]
        k = 0 if (u,v,0) in gdf_edges.index else next((kk for kk in G[u][v] if (u,v,kk) in gdf_edges.index),0)
        edges.append((u,v,k))
    route = gdf_edges.loc[edges].copy()

    # =============== 打印计时报告 ===============
    print("\n======= A★ Path‑finding Timing Report =======")
    print(f"Total elapsed              : {time.time()-t_global_start:.4f} s")
    print(f"Heuristic calls / time     : {h_calls} / {t_heuristic_accum:.4f} s")
    print(f"Cost calls      / time     : {td_calls} / {t_cost_accum:.4f} s")
    print(f"Neighbor visits  / time    : {neighbor_visits} / {t_neighbor_accum:.4f} s")
    print("=============================================\n")

    return route

def update_static_route(coef, start_time):
    """
    静态 A* ：阴影比例只取“起始时刻 start_time” 那一刻的快照，
    成本与动态版本同公式： sunny_dist(t0) + coef * edge_length
    返回 GeoDataFrame（蓝色路径）。
    """
    import heapq
    from math import hypot
    import osmnx as ox

    # ---------- 1. 起终点 ----------
    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        print("Start / End 未设置")
        return None

    orig_node = ox.distance.nearest_nodes(
        G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest_node = ox.distance.nearest_nodes(
        G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    # ---------- 2. 固定时间戳 ----------
    fixed_t = find_nearest_time(time_to_union.keys(), start_time)

    # ---------- 3. 启发函数 ----------
    def heuristic(n):
        lat1, lon1 = G.nodes[n]['y'], G.nodes[n]['x']
        lat2, lon2 = G.nodes[dest_node]['y'], G.nodes[dest_node]['x']
        try:                                             # OSMnx >=2
            dist = ox.distance.euclidean(lat1, lon1, lat2, lon2)
        except AttributeError:                           # OSMnx 1.x
            dist = ox.distance.euclidean_dist_vec(lat1, lon1, lat2, lon2)
        # 最坏情况：sunny_dist = length
        return dist * (1 + coef)

    # ---------- 4. A* 主循环 ----------
    open_pq = [(0.0, orig_node)]
    g_cost  = {orig_node: 0.0}
    parent  = {orig_node: None}

    while open_pq:
        _, cur = heapq.heappop(open_pq)
        if cur == dest_node:
            break

        for nbr, edict in G[cur].items():
            k0, attrs = next(iter(edict.items()))

            # ---- 4.1 取边长度 ----
            length = attrs.get('length')
            if length is None:
                geom = attrs.get('geometry')
                length = geom.length if geom is not None else 0.0
            if length == 0:
                continue

            # ---- 4.2 取固定时刻的阴影比例 -> sunny_dist ----
            shadow_ratio = precomputed.get((cur, nbr, k0, fixed_t), 1.0)  # 1.0 = 全阴影长度
            sunny_dist   = (1 - shadow_ratio) * length                    # ≡ 动态版公式

            #step_cost = sunny_dist + coef * length
            #cost = edge_length - coef * shadowed_length 改成这个咋样
            #cost = sunny_dist + coef * edge_length 这个好像才对？
            step_cost = sunny_dist+ coef * length
            tentative = g_cost[cur] + step_cost

            if tentative < g_cost.get(nbr, float('inf')):
                g_cost[nbr] = tentative
                parent[nbr] = (cur, k0)
                heapq.heappush(open_pq, (tentative + heuristic(nbr), nbr))

    # ---------- 5. 回溯 ----------
    if dest_node not in parent:
        print("Static A* 未找到路径")
        return None

    path_nodes = []
    n = dest_node
    while n is not None:
        path_nodes.append(n)
        n = parent[n][0] if parent[n] else None
    path_nodes.reverse()

    edge_triplets = [
        (path_nodes[i], path_nodes[i+1], parent[path_nodes[i+1]][1])
        for i in range(len(path_nodes)-1)
    ]
    return gdf_edges.loc[edge_triplets].copy()

# ---------------------------------------------------------------------------------------
# 设置 Slider 与按钮
# ---------------------------------------------------------------------------------------
initial_coef = 1
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.subplots_adjust(left=0.1, bottom=0.3)

ax_coef = plt.axes([0.1, 0.1, 0.65, 0.03])
coef_slider = Slider(
    ax=ax_coef,
    label='Shadow weight',
    valmin=-1.0,
    valmax=1.0,
    valinit=initial_coef,
    valstep=0.1
)
ax_coef.text(
    0.5, -1.2,
    'Weight is from -1 (ride under SUN) to 1 (ride under SHADOW). ',
    ha='center',
    va='center',
    transform=ax_coef.transAxes
)

ax_button_update = plt.axes([0.8, 0.05, 0.1, 0.075])
button_update = Button(ax_button_update, 'Update Route')

def update_route(event):
    coef_val = coef_slider.val
    new_route_gdf = update_cool_route(coef_val, start_time)

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
    plt.legend(handles=[shortest_route_legend, wanted_route_legend,], loc='upper right',
               bbox_to_anchor=(-2, 1.05))
    plt.draw()
    print("wanted route:")
    calculate_shadow_stats(new_route_gdf, time_to_union, start_time, coef=coef_slider.val)

button_update.on_clicked(update_route)

# ---------------------------------------------------------------------------------------
# Generate Path按钮
# ---------------------------------------------------------------------------------------
ax_button_generate = plt.axes([0.65, 0.15, 0.1, 0.075])
ax_button_generate.text(
    -4.5, 0.65,
    'Please click on the map to select the start and end points ',
    ha='center',
    va='center',
    transform=ax_button_generate.transAxes
)
button_generate = Button(ax_button_generate, 'Generate Path')

def generate_path(event):
    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        print("请先在地图上点击选择起点和终点。")
        return

    orig_node = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest_node = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    starttimeshortpath = time.time()
    route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
    endtimeshortpath = time.time()
    timeshortpath = endtimeshortpath - starttimeshortpath
    print("计算最短路径的时间", timeshortpath)

    route_edges = [(route[i], route[i+1], 0) for i in range(len(route) - 1)]
    route_gdf = gdf_edges.loc[route_edges]

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Shortest Bike Route':
            artist.remove()
    route_gdf.plot(ax=ax, color='red', linewidth=2, label='Shortest Bike Route')

    coef_val = coef_slider.val
    new_route_gdf = update_cool_route(coef_val, start_time)

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    print("shortest route:")
    calculate_shadow_stats(route_gdf, time_to_union, start_time, coef=coef_slider.val)

    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')

    plt.legend(handles=[shortest_route_legend, wanted_route_legend,],
               loc='upper right', bbox_to_anchor=(-2, 1.05))
    plt.draw()
    print("wanted route:")
    calculate_shadow_stats(new_route_gdf, time_to_union, start_time, coef=coef_slider.val)
    # ① 生成静态 A*
    static_route_gdf = update_static_route(coef_slider.val, start_time)
    if static_route_gdf is not None:
        # 先删旧蓝线
        for a in ax.lines + ax.collections:
            if a.get_label() == 'Static A* Route':
                a.remove()
        # ✅ 加上偏移
        from shapely.affinity import translate
        static_route_gdf['geometry'] = static_route_gdf.geometry.apply(
            lambda g: translate(g, xoff=-6.0, yoff=6.0)
        )
        # 再绘图
        static_route_gdf.plot(ax=ax, color='orange', linewidth=2, label='Static A* Route')

        print("static A* route:")
        calculate_shadow_stats(static_route_gdf,
                               time_to_union, start_time, coef=coef_slider.val)


button_generate.on_clicked(generate_path)

# ---------------------------------------------------------------------------------------
# 新增：清除起点和终点按钮
# ---------------------------------------------------------------------------------------
ax_button_clear = plt.axes([0.8, 0.15, 0.1, 0.075])
button_clear = Button(ax_button_clear, 'Clear Points')
def clear_points(event):
    global click_count, origin_point_wgs84, destination_point_wgs84
    global origin_marker, destination_marker

    click_count = 0
    origin_point_wgs84 = None
    destination_point_wgs84 = None

    if origin_marker is not None:
        origin_marker.remove()
        origin_marker = None
    if destination_marker is not None:
        destination_marker.remove()
        destination_marker = None

    for artist in ax.lines + ax.collections:
        if artist.get_label() in ['Shortest Bike Route', 'Wanted Bike Route', 'Origin', 'Destination']:
            artist.remove()
    plt.draw()

button_clear.on_clicked(clear_points)

# ---------------------------------------------------------------------------------------
# North Arrow
# ---------------------------------------------------------------------------------------
from matplotlib.offsetbox import AnchoredText
north_arrow = AnchoredText('↑ North', loc='upper left', pad=0, prop=dict(size=14), frameon=False)
ax.add_artist(north_arrow)

plt.show()
