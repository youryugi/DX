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

# ---------------------------------------------------------------------------------------
# 1) 读取离线计算好的 (u,v,k, time) -> shadow_ratio
# 注意：请确认与下面的 G、时间切分一致
# ---------------------------------------------------------------------------------------
with open("edge_shadow_ratios_20241205_1300_1400_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl", "rb") as f:
    precomputed = pickle.load(f)
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

shadow_file = r"shadows_20241205_1300_1400_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl"
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
wanted_route_legend   = Line2D([0], [0], color='green', linewidth=2, label='Wanted Bike Route')
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
          f"非阴影下路程: {non_shadow_distance:.2f} m, "
          f"总路程: {total_distance:.2f} m")
    print("计算了", intersection_counter)

# ---------------------------------------------------------------------------------------
# 读取 time_to_union（每个时刻的阴影区域），并找 nearest_time
# ---------------------------------------------------------------------------------------

with open(shadow_file, 'rb') as f:
    time_to_union = pickle.load(f)

def find_nearest_time(keys, target):
    return min(keys, key=lambda t: abs(t - target))

date_time = datetime(2024, 12, 5, 13, 10, tzinfo=timezone(timedelta(hours=9)))
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
    Line2D([0], [0], color='green', linewidth=2, label='Wanted Route'),
]
plt.title("Roads, buildings, and shadows     ↑North", fontsize=16)
plt.legend(handles=legend_handles, loc='lower left', fontsize=bigfontsize)
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
def update_cool_route(coef, start_time, sample_interval=300):
    intersection_counter = 0
    intersection_total_time = 0.0
    """
    实现更加严谨的“多状态时变 Dijkstra”：
      - 同一节点可在不同时间到达时，同时保留多条可能路径，
      - 不再简单丢弃“时间更晚”或“cost更大”的状态，
      - 只有当一个状态被另一个在时间或cost上严格支配(dominates)时，才会丢弃。

    参数:
    - coef: 用户Slider指定的阴影系数(-1~1等)
    - start_time: datetime, 出发时间
    - sample_interval: 分段采样时，每段的秒数
    返回:
    - new_cool_route_gdf: 找到的最优时变路线对应的GeoDataFrame(带1.5m偏移)，若无路则返回None
    """
    import heapq
    from datetime import timedelta

    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        return None

    orig_node = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest_node = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    # =============== 2. 定义分段阴影计算函数 ========================
    def time_dependent_cost(u, v, k, arrival_dt):
        """
        给定边 (u, v, k) 以及到达此边的时刻 arrival_dt，
        分段采样计算在该时段穿越此边的 cost + 耗时。
        返回 (edge_cost, edge_duration)
        """
        if (u, v, k) not in gdf_edges.index:
            return float('inf'), float('inf')

        edge_geom = gdf_edges.loc[(u, v, k), 'geometry']
        edge_length = edge_geom.length
        speed = 10 * 1000 / 3600  # 10km/h
        edge_duration = edge_length / speed

        cost_accumulate = 0.0
        temp_time = arrival_dt
        remaining_time = edge_duration

        while remaining_time > 0:
            dt = min(sample_interval, remaining_time)
            mid_time = temp_time + timedelta(seconds=dt / 2)

            # 找最近阴影时刻
            nearest_t = find_nearest_time(time_to_union.keys(), mid_time)
            shadow_poly = time_to_union[nearest_t] if nearest_t else None

            if shadow_poly:
                nonlocal intersection_counter, intersection_total_time
                intersection_counter += 1
                t1 = time.time()

                # ==============================
                # 这里使用预先计算好的 ratio
                ratio = precomputed.get((u, v, k, nearest_t), 0.0)
                shadow_len = ratio * edge_length
                # ==============================

                t2 = time.time()
                intersection_total_time += (t2 - t1)
            else:
                shadow_len = 0.0

            shadow_ratio = shadow_len / edge_length if edge_length > 0 else 0.0
            sunny_dist = edge_length * (1 - shadow_ratio)

            cost_part = (sunny_dist + coef * edge_length) * (dt / edge_duration)
            cost_accumulate += cost_part

            temp_time += timedelta(seconds=dt)
            remaining_time -= dt

        return cost_accumulate, edge_duration

    # =============== 3. 定义支配判定函数 ==========================
    def is_dominated(new_t, new_cost, states):
        for (t_s, c_s) in states:
            if t_s <= new_t and c_s <= new_cost and (t_s < new_t or c_s < new_cost):
                return True
        return False

    def remove_dominated(new_t, new_cost, states):
        non_dominated = []
        for (t_s, c_s) in states:
            if new_t <= t_s and new_cost <= c_s and (new_t < t_s or new_cost < c_s):
                continue
            non_dominated.append((t_s, c_s))
        return non_dominated

    # =============== 4. 优先队列 + 多状态管理 ======================
    starttimewantedpath = time.time()
    pq = []
    best_states = {}
    predecessor = {}

    init_time_s = 0.0
    init_cost = 0.0
    best_states[orig_node] = [(init_time_s, init_cost)]
    predecessor[(orig_node, init_time_s, init_cost)] = None
    heapq.heappush(pq, (init_cost, orig_node, init_time_s))

    found_path = False
    dest_best_state = None

    while pq:
        current_cost, current_node, current_time_s = heapq.heappop(pq)

        if current_node not in best_states:
            continue
        if (current_time_s, current_cost) not in best_states[current_node]:
            continue

        if current_node == dest_node:
            found_path = True
            dest_best_state = (current_node, current_time_s, current_cost)
            break

        arrival_dt = start_time + timedelta(seconds=current_time_s)

        if current_node not in G:
            continue

        for neighbor, edges_dict in G[current_node].items():
            for k, edge_attrs in edges_dict.items():
                edge_cost, edge_dur = time_dependent_cost(current_node, neighbor, k, arrival_dt)
                if edge_cost == float('inf'):
                    continue

                new_time_s = current_time_s + edge_dur
                new_cost = current_cost + edge_cost

                if neighbor not in best_states:
                    best_states[neighbor] = [(new_time_s, new_cost)]
                    predecessor[(neighbor, new_time_s, new_cost)] = (current_node, current_time_s, current_cost)
                    heapq.heappush(pq, (new_cost, neighbor, new_time_s))
                else:
                    if not is_dominated(new_time_s, new_cost, best_states[neighbor]):
                        updated_list = remove_dominated(new_time_s, new_cost, best_states[neighbor])
                        updated_list.append((new_time_s, new_cost))
                        best_states[neighbor] = updated_list

                        predecessor[(neighbor, new_time_s, new_cost)] = (current_node, current_time_s, current_cost)
                        heapq.heappush(pq, (new_cost, neighbor, new_time_s))

    endtimewantedpath = time.time()
    timewantedpath = endtimewantedpath - starttimewantedpath
    print("计算想要的路径的时间(严谨版时变) =", timewantedpath, "秒")

    if not found_path or not dest_best_state:
        print("未找到可行的时变路线。")
        return None

    final_node, final_time_s, final_cost = dest_best_state
    route_nodes = []
    cur_state = (final_node, final_time_s, final_cost)
    while cur_state is not None:
        n, t_s, c = cur_state
        route_nodes.append(n)
        cur_state = predecessor.get(cur_state, None)
    route_nodes.reverse()

    # =============== 6. 将节点序列转为边序列，提取geometry，偏移 =================
    new_cool_route_edges = []
    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]

        found_k = None
        if (u, v, 0) in gdf_edges.index:
            found_k = 0
        else:
            if v in G[u]:
                for possible_k in G[u][v].keys():
                    if (u, v, possible_k) in gdf_edges.index:
                        found_k = possible_k
                        break
        if found_k is not None:
            new_cool_route_edges.append((u, v, found_k))

    if not new_cool_route_edges:
        print("回溯时没找到对应的边数据，无法绘制。")
        return None

    new_cool_route_gdf = gdf_edges.loc[new_cool_route_edges].copy()
    from shapely.affinity import translate
    new_cool_route_gdf['geometry'] = new_cool_route_gdf.geometry.apply(lambda g: translate(g, xoff=1.5, yoff=1.5))

    print(f"最终时变路线: 节点数={len(route_nodes)}, cost={final_cost:.3f}, "
          f"到达时间=+{final_time_s:.1f}秒(相对于出发时刻)")
    print(f"本次路径计算共执行了 {intersection_counter} 次 intersection() 操作")
    print(f"所有 intersection() 总耗时: {intersection_total_time:.4f} 秒")

    return new_cool_route_gdf

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
