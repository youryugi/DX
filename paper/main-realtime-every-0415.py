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
#第k圈中阴影不变
#factor是允许的时间增加的长度，，，是不是要和coef联系在一起
# 在代码开头处定义图例句柄
shortest_route_legend = Line2D([0], [0], color='red', linewidth=2, label='Shortest Bike Route')
wanted_route_legend = Line2D([0], [0], color='green', linewidth=2, label='Wanted Bike Route')
bigfontsize=14
# 路径设置
bldg_gml_files = [
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357451_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357452_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357453_bldg_6697_op.gml"
]

# pkl保存路径
pkl_path = r"C:\Users\79152\Desktop\OthersProgramme\DX\paper\bldg_merged_gdf.pkl"

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

# 直接用 bldg_merged_gdf
print(bldg_merged_gdf.head())
# -------------------------
starttime2=time.time()
road_gml_files = [
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357451_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357452_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357453_tran_6697_op.gml"
    # 以及其它所有 road gml
]

road_gdf_list = [gpd.read_file(file) for file in road_gml_files]
merged_road_gdf = pd.concat(road_gdf_list, ignore_index=True)
endtime2=time.time()
readtrantime=endtime2-starttime2
print("读取tran的时间",readtrantime)


building_gdf = bldg_merged_gdf
road_gdf = merged_road_gdf

# 确保投影为EPSG:6669
if building_gdf.crs.to_epsg() != 6669:
    building_gdf = building_gdf.to_crs(epsg=6669)
if road_gdf.crs.to_epsg() != 6669:
    road_gdf = road_gdf.to_crs(epsg=6669)

def calculate_shadow_stats(route_gdf, time_to_union, start_time, coef, sample_interval=60):
    speed = 10 * 1000 / 3600  # 10km/h -> m/s

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

    print(f"阴影下路程: {shadow_distance:.2f} m, 非阴影下路程: {non_shadow_distance:.2f} m, 总路程: {total_distance:.2f} m")


import pickle
# 指定时间
date_time = datetime(2024, 12, 5, 13, 10, tzinfo=timezone(timedelta(hours=9)))
# 加载提前计算好的阴影文件（里面是 time_to_union = {datetime: shapely geometry}）
shadow_file = r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\shadows_20241205_1300_1400_1min.pkl"
with open(shadow_file, 'rb') as f:
    time_to_union = pickle.load(f)
# 找最近时刻
def find_nearest_time(keys, target):
    return min(keys, key=lambda t: abs(t - target))

nearest_time = find_nearest_time(time_to_union.keys(), date_time)
shadow_union = time_to_union[nearest_time]

if shadow_union is None:
    print("该时刻太阳在地平线以下，或无有效阴影。")
    exit()

# 构造 shadow_gdf 以便你后续绘图
shadow_gdf = gpd.GeoDataFrame(geometry=[shadow_union], crs='EPSG:6669')  # 用你建筑的坐标系

# -------------------------
# 准备绘图
# -------------------------
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

road_gdf.plot(ax=ax, color='yellow', alpha=0.5,label='road')
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='shadow')
building_gdf.plot(ax=ax, color='lightblue',  label='building')

legend_handles = [
    Patch(facecolor='yellow', label='Road'),
    Patch(facecolor='lightblue', label='Building'),
    Patch(facecolor='gray', edgecolor='gray', label='Shadow'),
    Line2D([0], [0], marker='o', color='blue', markersize=8, linestyle='None', label='Start Point'),
    Line2D([0], [0], marker='o', color='magenta', markersize=8, linestyle='None', label='End Point'),
    Line2D([0], [0], color='red', linewidth=2, label='Shortest Route'),
    Line2D([0], [0], color='green', linewidth=2, label='Wanted Route'),
]
plt.title("Roads, buildings, and shadows     ↑North", fontsize=16)
plt.legend(handles=legend_handles, loc='lower left',fontsize=bigfontsize)
plt.xlabel("X (m)", fontsize=bigfontsize)
plt.ylabel("Y (m)", fontsize=bigfontsize)
plt.xticks(fontsize=bigfontsize)
plt.yticks(fontsize=bigfontsize)

plt.grid(True)

# -------------------------
# 获取路网和路线规划
# -------------------------
# 在下载OSM数据时需要WGS84
building_gdf_wgs84 = building_gdf.to_crs(epsg=4326)
building_bounds_wgs84 = building_gdf_wgs84.total_bounds  # (minx, miny, maxx, maxy)

# bbox顺序：(北, 南, 东, 西)
bbox = (building_bounds_wgs84[3], building_bounds_wgs84[1],
        building_bounds_wgs84[2], building_bounds_wgs84[0])
#下载osm或者是加载已经有的
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

# 将图转换为GeoDataFrame，并投影为EPSG:6669以计算长度和阴影覆盖
gdf_edges = ox.graph_to_gdfs(G, nodes=False)
if gdf_edges.crs.to_epsg() != 4326:
    gdf_edges = gdf_edges.to_crs(epsg=4326)
# 转换为EPSG:6669
gdf_edges = gdf_edges.to_crs(epsg=6669)

# 合并阴影用于后续计算
shadow_union = unary_union(shadow_gdf.geometry)

# ---------------------
# 准备交互选点和路径更新功能
# ---------------------
transformer_to_wgs84 = Transformer.from_crs(6669, 4326, always_xy=True)

click_count = 0
origin_point_wgs84 = None
destination_point_wgs84 = None
origin_marker = None
destination_marker = None

# 手动输入模式，用于程序复现（True开启，False关闭）
manual_input_mode = True

# 预设的起点和终点（WGS84，经纬度格式）
manual_origin_point_wgs84 = (34.632734242239636, 135.51493454131852)  # 大阪
manual_destination_point_wgs84 = (34.63049080065106, 135.54547444776205)  #
# 已选择起点: (lat=34.632734242239636, lon=135.51493454131852)
# 已选择终点: (lat=34.63049080065106, lon=135.54547444776205)
# ------------------- 坐标系相关 -------------------
proj_crs = 'EPSG:6669'  # 日本 GML 文件的坐标系

from pyproj import Transformer

# 正向（地图坐标 -> WGS84经纬度）
transformer_to_wgs84 = Transformer.from_crs(proj_crs, 'EPSG:4326', always_xy=True)

# 反向（WGS84经纬度 -> 地图坐标）
transformer_from_wgs84 = Transformer.from_crs('EPSG:4326', proj_crs, always_xy=True)


def on_map_click(event):
    global click_count, origin_point_wgs84, destination_point_wgs84, origin_marker, destination_marker

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
start_time=date_time


def update_cool_route(coef, start_time, ring_minutes=5, factor=1.3):
    """
    使用“第k圈阴影不变”的思路，将预处理(分圈/一次intersection)和静态最短路径一起写在本函数里。

    参数：
    - coef: 阴影系数(从-1到1等)
    - start_time: datetime, 假设的出发时间
    - ring_minutes: 每个圈对应的分钟数(默认5)
    - factor: 额外乘的系数(>1表示到达更晚, <1表示更快)

    返回：
    - route_gdf: 路线的GeoDataFrame(已做1.5m平移)，若无起终点或无路径则返回None

    注意：这会覆盖掉你原先的时变搜索逻辑(多状态Dijkstra)，
         改为基于“圈+固定阴影”的静态近似方法。
    """
    import math
    import numpy as np
    from shapely.ops import unary_union
    from datetime import timedelta

    # 如果起终点未指定，直接返回None
    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        print("尚未选择起点和终点，无法生成路线。")
        return None

    print(f"【第k圈+静态阴影】计算路线: coef={coef}, ring_minutes={ring_minutes}, factor={factor}")
    starttimewantedroute=time.time()
    # 1) 找到起终点对应的图节点
    orig_node = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest_node = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    # 如果orig_node或dest_node为None，返回
    if orig_node is None or dest_node is None:
        print("无法找到对应起终点node。")
        return None

    # 2) 用单源最短路算各节点到起点的距离
    dist_to_origin = nx.single_source_dijkstra_path_length(G, source=orig_node, weight='length')

    # 3) 准备对 gdf_edges 做“圈”的阴影比计算
    # 为安全，复制一份
    gdf_local = gdf_edges.copy()

    # 计算 speed  (默认 10km/h)
    # ring_dist = ring_minutes * 60 * (10km/h转m/s)= ring_minutes * 60 * (10*1000/3600)
    base_speed_m_s = 10 * 1000 / 3600.0
    ring_dist = ring_minutes * 60.0 * base_speed_m_s

    # 定义找最近阴影时刻
    def find_nearest_time(keys, target):
        return min(keys, key=lambda t: abs(t - target))

    # 4) 遍历每条边，分配 ring_k, intersection
    ring_indices = []
    ring_shadow_ratios = []

    for idx, row in gdf_local.iterrows():
        (u, v, k) = idx
        edge_geom = row.geometry
        if (u not in dist_to_origin) or (v not in dist_to_origin):
            # 可能不连通
            ring_indices.append(-1)
            ring_shadow_ratios.append(0.0)
            continue

        d_u = dist_to_origin[u]
        d_v = dist_to_origin[v]
        avg_dist = (d_u + d_v) / 2.0

        # 计算 ring_k = floor((avg_dist / ring_dist) * factor)
        ring_k = int(math.floor((avg_dist / ring_dist) * factor))
        if ring_k < 0:
            ring_k = 0

        # 代表时间 rep_time = start_time + ring_k * ring_minutes
        # 若你想再额外乘 factor在时间上，也可以:
        rep_time = start_time + timedelta(minutes=ring_k * ring_minutes * factor)
        #rep_time = start_time + timedelta(minutes=ring_k * ring_minutes)#没有额外乘factor

        # 找 nearest 阴影
        if len(time_to_union) == 0:
            # 没阴影信息
            ring_indices.append(ring_k)
            ring_shadow_ratios.append(0.0)
            continue

        nearest_t = find_nearest_time(time_to_union.keys(), rep_time)
        shadow_poly = time_to_union[nearest_t] if nearest_t else None

        if shadow_poly and not shadow_poly.is_empty:
            inters = edge_geom.intersection(shadow_poly)
            shadow_len = inters.length if inters and not inters.is_empty else 0.0
            ratio = shadow_len / edge_geom.length if edge_geom.length > 0 else 0.0
        else:
            ratio = 0.0

        ring_indices.append(ring_k)
        ring_shadow_ratios.append(ratio)

    # 把 ring信息存到临时列
    gdf_local["ring_index"] = ring_indices
    gdf_local["ring_shadow_ratio"] = ring_shadow_ratios

    # 5) 计算 cost = (1-shadow_ratio)*length + coef*(shadow_ratio*length)
    ring_costs = []
    for idx, row in gdf_local.iterrows():
        length = row.geometry.length
        sr = row["ring_shadow_ratio"]
        cval = length * (1 - sr) + coef * (length * sr)
        ring_costs.append(cval)
    gdf_local["ring_cost"] = ring_costs

    # 6) 把 ring_cost 写回图 G[u][v][k]['ring_cost']
    for (u, v, k), cost_val in zip(gdf_local.index, gdf_local["ring_cost"]):
        if (u in G) and (v in G[u]) and (k in G[u][v]):
            G[u][v][k]["ring_cost"] = cost_val

    # 7) 用普通最短路径
    try:
        route_nodes = nx.shortest_path(G, source=orig_node, target=dest_node, weight='ring_cost')
    except nx.NetworkXNoPath:
        print("无可行路径 (圈阴影近似)。")
        return None

    endtimewanted=time.time()
    print("生成wanted route需要的时间",endtimewanted-starttimewantedroute)
    # 8) 把节点序列转成边序列
    route_edges = []
    for i in range(len(route_nodes) - 1):
        n1 = route_nodes[i]
        n2 = route_nodes[i + 1]
        found_k = None
        if (n1, n2, 0) in gdf_local.index:
            found_k = 0
        else:
            # 多重图,检查key
            if (n2 in G[n1]):
                for possible_k in G[n1][n2].keys():
                    if (n1, n2, possible_k) in gdf_local.index:
                        found_k = possible_k
                        break
        if found_k is not None:
            route_edges.append((n1, n2, found_k))

    if not route_edges:
        print("找不到对应边 (圈阴影近似)。")
        return None

    # 9) 提取路线gdf，并平移(1.5,1.5)
    route_gdf = gdf_local.loc[route_edges].copy()
    from shapely.affinity import translate
    route_gdf["geometry"] = route_gdf.geometry.apply(lambda g: translate(g, xoff=1.5, yoff=1.5))

    print(f"【第k圈+静态阴影】路线计算完成: 边数={len(route_edges)}")
    return route_gdf


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
# 在图像上方添加提示文字

def update_route(event):
    coef_val = coef_slider.val
    new_route_gdf = update_cool_route(coef_val,start_time)

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
    # 在绘制完成后手动更新图例
    plt.legend(handles=[shortest_route_legend, wanted_route_legend,], loc='upper right',          # 图例框锚点在图例的左上角
    bbox_to_anchor=(-2, 1.05))
    plt.draw()
    print("wanted route:")
    calculate_shadow_stats(new_route_gdf, time_to_union, start_time, coef=coef_slider.val)


button_update.on_clicked(update_route)

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

    # 最短路径
    starttimeshortpath=time.time()
    route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
    endtimeshortpath=time.time()
    timeshortpath=endtimeshortpath-starttimeshortpath
    print("计算最短路径的时间",timeshortpath)
    route_edges = [(route[i], route[i+1], 0) for i in range(len(route)-1)]
    route_gdf = gdf_edges.loc[route_edges]

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Shortest Bike Route':
            artist.remove()

    route_gdf.plot(ax=ax, color='red', linewidth=2, label='Shortest Bike Route')

    # Wanted Bike Route
    coef_val = coef_slider.val
    new_route_gdf = update_cool_route(coef_val, start_time)

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()
    print("shortest route:")
    calculate_shadow_stats(route_gdf, time_to_union, start_time, coef=coef_slider.val)

    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
    plt.legend(handles=[shortest_route_legend, wanted_route_legend,], loc='upper right',          # 图例框锚点在图例的左上角
    bbox_to_anchor=(-2, 1.05))
    plt.draw()
    print("wanted route:")
    calculate_shadow_stats(new_route_gdf, time_to_union, start_time, coef=coef_slider.val)

    # ==================增加在coef在0-10的情况下，对所有
    results = []
    coef_values = np.arange(0, 1.1, 0.1)

    orig_node = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest_node = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    # 计算路径长度函数（依旧可以用）
    def calc_length(gdf):
        total = 0
        sunny = 0
        shadow = 0
        for idx, row in gdf.iterrows():
            edge_geom = row.geometry
            length = edge_geom.length
            inter = edge_geom.intersection(shadow_union)
            shadow_len = inter.length if not inter.is_empty else 0
            sunny_len = length - shadow_len if length > 0 else 0
            total += length
            sunny += sunny_len
            shadow += shadow_len
        return total, sunny, shadow

    # 计算不同 coef 下的路径
    for coef in coef_values:
        route_gdf = update_cool_route(coef, start_time)  # 严谨版的路径直接返回gdf

        if route_gdf is None:  # 防止没找到路径
            continue

        total, sunny, shadow = calc_length(route_gdf)
        results.append([coef, total, sunny, shadow])

    # 最短路径部分依旧用 length
    shortest_route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
    shortest_edges = [(shortest_route[i], shortest_route[i + 1], 0) for i in range(len(shortest_route) - 1)]
    shortest_gdf = gdf_edges.loc[shortest_edges]

    shortest_total, shortest_sunny, shortest_shadow = calc_length(shortest_gdf)

    results.append(['Shortest', shortest_total, shortest_sunny, shortest_shadow])

    df = pd.DataFrame(results, columns=[
        'coef', 'total_length', 'sunny_length', 'shadow_length'
    ])

    df.to_excel('route_length_analysis_everyminute_big.xlsx', index=False)


# ============================================================================================


button_generate.on_clicked(generate_path)

# 新增：清除起点和终点的按钮
ax_button_clear = plt.axes([0.8, 0.15, 0.1, 0.075])
button_clear = Button(ax_button_clear, 'Clear Points')
def clear_points(event):
    global click_count, origin_point_wgs84, destination_point_wgs84, origin_marker, destination_marker

    # 重置点击计数和点信息
    click_count = 0
    origin_point_wgs84 = None
    destination_point_wgs84 = None

    # 清除标记
    if origin_marker is not None:
        origin_marker.remove()
        origin_marker = None
    if destination_marker is not None:
        destination_marker.remove()
        destination_marker = None

    # 清除路线（可选）
    for artist in ax.lines + ax.collections:
        if artist.get_label() in ['Shortest Bike Route', 'Wanted Bike Route', 'Origin', 'Destination']:
            artist.remove()

    # 不再调用 plt.legend(...)
    # plt.legend(handles=legend_handles, loc='upper right')
    plt.draw()

button_clear.on_clicked(clear_points)
#增加north
from matplotlib.offsetbox import AnchoredText
north_arrow = AnchoredText('↑ North', loc='upper left', pad=0, prop=dict(size=14), frameon=False)
ax.add_artist(north_arrow)
plt.show()
