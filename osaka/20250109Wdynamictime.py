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

# -------------------------
# 全局图例句柄
# -------------------------
shortest_route_legend = Line2D([0], [0], color='red', linewidth=2, label='Shortest Bike Route')
wanted_route_legend = Line2D([0], [0], color='green', linewidth=2, label='Wanted Bike Route')

# -------------------------
# 读多个建筑物文件并合并
# -------------------------
bldg_gml_files = [
    r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357451_bldg_6697_op.gml",
    r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357452_bldg_6697_op.gml",
    r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357453_bldg_6697_op.gml"
]
bldg_gdf_list = [gpd.read_file(file) for file in bldg_gml_files]
bldg_merged_gdf = pd.concat(bldg_gdf_list, ignore_index=True)

# 读多个道路文件并合并
road_gml_files = [
    r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\tran\51357451_tran_6697_op.gml",
    r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\tran\51357452_tran_6697_op.gml",
    r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\tran\51357453_tran_6697_op.gml"
]
road_gdf_list = [gpd.read_file(file) for file in road_gml_files]
merged_road_gdf = pd.concat(road_gdf_list, ignore_index=True)

building_gdf = bldg_merged_gdf
road_gdf = merged_road_gdf

# 确保投影为EPSG:6669
if building_gdf.crs.to_epsg() != 6669:
    building_gdf = building_gdf.to_crs(epsg=6669)
if road_gdf.crs.to_epsg() != 6669:
    road_gdf = road_gdf.to_crs(epsg=6669)

# -------------------------
# 定义基础参数：城市信息、初始时间、速度等
# -------------------------
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo",
                    latitude=34.6937, longitude=135.5023)
# 假设出发时间：2024-12-05 13:10 (含时区信息)
base_time = datetime(2024, 12, 5, 13, 10, tzinfo=timezone(timedelta(hours=9)))

# 自行车速度 (用于更深入模拟可做分段式，但这里主要用来提示)
bike_speed_km_h = 10  # 10 km/h

# -------------------------
# 定义一个函数，用于给定时间重新计算太阳矢量、阴影等
# -------------------------
def recalc_sun_and_shadow(building_gdf, current_time):
    """
    给定 GeoDataFrame(building_gdf) 和当前时间(current_time)，
    计算新的太阳高度角、方位角，进而计算建筑物阴影。
    返回: (building_gdf_with_shadow, shadow_union)
    """
    # 计算太阳高度角和方位角
    sol_elev = elevation(city.observer, current_time)
    sol_azim = azimuth(city.observer, current_time)

    if sol_elev <= 0:
        # 如果太阳在地平线以下，返回 None，让外部判断
        return None, None

    # 计算太阳矢量
    sun_vec = np.array([
        np.cos(np.radians(sol_elev)) * np.sin(np.radians(sol_azim)),
        np.cos(np.radians(sol_elev)) * np.cos(np.radians(sol_azim)),
        np.sin(np.radians(sol_elev))
    ])

    # 找到高度列（或设置默认高度）
    height_col = None
    for col in building_gdf.columns:
        if 'height' in col.lower():
            height_col = col
            break
    if height_col is not None:
        building_gdf[height_col] = building_gdf[height_col].fillna(3)
    else:
        height_col = 'default_height'
        building_gdf[height_col] = 3.0

    # 计算阴影的函数
    def shadow_using_lines(geometry, height):
        if sun_vec[2] <= 0:
            return None
        polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
        shadow_lines = []

        for poly in polygons:
            base_coords = [(x, y) for x, y, *rest in poly.exterior.coords]
            shadow_coords = [
                (
                    x - height / np.tan(np.radians(sol_elev)) * sun_vec[0],
                    y - height / np.tan(np.radians(sol_elev)) * sun_vec[1]
                )
                for x, y in base_coords
            ]
            for base, shdw in zip(base_coords, shadow_coords):
                shadow_lines.append(LineString([base, shdw]))

        union_lines = unary_union(shadow_lines)
        shadow_polygon = union_lines.convex_hull
        return shadow_polygon

    # 新建一列 'shadow'，存放新计算的阴影几何
    bdf = building_gdf.copy()  # 不要修改原 DataFrame
    bdf['shadow'] = bdf.apply(
        lambda row: shadow_using_lines(row.geometry, row[height_col]), axis=1
    )
    # 合并所有阴影
    shadow_gdf = bdf.dropna(subset=['shadow']).set_geometry('shadow')
    shadow_union = unary_union(shadow_gdf.geometry)

    return bdf, shadow_union

# -------------------------
# 初始化：先计算一次阴影(使用 base_time)
# -------------------------
building_gdf_with_shadow, shadow_union = recalc_sun_and_shadow(building_gdf, base_time)
if (building_gdf_with_shadow is None) or (shadow_union is None):
    print("初始时刻太阳在地平线以下或阴影计算有误，停止。")
    exit()

# -------------------------
# 准备绘图
# -------------------------
plt.rcParams['font.family'] = 'SimHei'
fig, ax = plt.subplots(figsize=(12, 8))

bounds = building_gdf_with_shadow.total_bounds
buffer = 10
x_min, y_min, x_max, y_max = bounds
x_min -= buffer
y_min -= buffer
x_max += buffer
y_max += buffer

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# 绘制道路、阴影、建筑物
road_gdf.plot(ax=ax, color='yellow', alpha=0.5, label='road')
shadow_gdf_initial = building_gdf_with_shadow.dropna(subset=['shadow']).set_geometry('shadow')
shadow_gdf_initial.plot(ax=ax, color='gray', alpha=0.5, label='shadow')
building_gdf_with_shadow.plot(ax=ax, color='lightblue', label='building')

legend_handles = [
    Patch(facecolor='yellow', label='Road'),
    Patch(facecolor='lightblue', label='Building'),
    Patch(facecolor='gray', edgecolor='black', label='Shadow'),
]
plt.title("Roads, buildings, and shadows     ↑North", fontsize=16)
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)

# -------------------------
# 获取路网和路线规划的准备
# -------------------------
# OSM 下载需要 WGS84
building_gdf_wgs84 = building_gdf_with_shadow.to_crs(epsg=4326)
building_bounds_wgs84 = building_gdf_wgs84.total_bounds  # (minx, miny, maxx, maxy)

# bbox 顺序：(北, 南, 东, 西)
bbox = (building_bounds_wgs84[3], building_bounds_wgs84[1],
        building_bounds_wgs84[2], building_bounds_wgs84[0])
print('开始下载OSM数据...')
print('bbox:', bbox)
G = ox.graph_from_bbox(north=bbox[0], south=bbox[1], east=bbox[2], west=bbox[3], network_type="bike")
print('下载完成')
print(f"节点数量: {len(G.nodes)}")
print(f"边数量: {len(G.edges)}")

# 将图转换为GeoDataFrame，并投影为EPSG:6669以计算长度和阴影覆盖
gdf_edges = ox.graph_to_gdfs(G, nodes=False)
if gdf_edges.crs.to_epsg() != 4326:
    gdf_edges = gdf_edges.to_crs(epsg=4326)
gdf_edges = gdf_edges.to_crs(epsg=6669)

# ---------------------
# 用于后续路线计算的函数
# ---------------------
transformer_to_wgs84 = Transformer.from_crs(6669, 4326, always_xy=True)

click_count = 0
origin_point_wgs84 = None
destination_point_wgs84 = None
origin_marker = None
destination_marker = None

def on_map_click(event):
    global click_count, origin_point_wgs84, destination_point_wgs84
    global origin_marker, destination_marker

    if event.inaxes != ax:
        return

    x_coord, y_coord = event.xdata, event.ydata
    lon, lat = transformer_to_wgs84.transform(x_coord, y_coord)

    if click_count == 0:
        # 第一次点击，设置起点
        origin_point_wgs84 = (lat, lon)
        if origin_marker is not None:
            origin_marker.remove()
        origin_marker = ax.plot(x_coord, y_coord, marker='o', color='blue',
                                markersize=10, label='Origin')[0]
        plt.draw()
        click_count += 1
        print(f"已选择起点: (lat={lat}, lon={lon})")
    elif click_count == 1:
        # 第二次点击，设置终点
        destination_point_wgs84 = (lat, lon)
        if destination_marker is not None:
            destination_marker.remove()
        destination_marker = ax.plot(x_coord, y_coord, marker='o', color='magenta',
                                     markersize=10, label='Destination')[0]
        plt.draw()
        click_count += 1
        print(f"已选择终点: (lat={lat}, lon={lon})")

fig.canvas.mpl_connect('button_press_event', on_map_click)

def update_cool_route(coef, shadow_union_local):
    """
    根据系数coef、阴影合并几何shadow_union_local，对路网计算新的 cool_weight，
    并返回“想要的”自行车路径 (Wanted Bike Route) 对应的 gdf_edges。
    若尚未选择起终点，返回 None。
    """
    shadow_lengths = []
    for idx, row in gdf_edges.iterrows():
        edge_geom = row.geometry
        edge_length = edge_geom.length
        intersection_geom = edge_geom.intersection(shadow_union_local)
        shadowed_length = intersection_geom.length if not intersection_geom.is_empty else 0
        shadow_ratio = shadowed_length / edge_length if edge_length > 0 else 0

        # cost = edge_length * (1 - coef * shadow_ratio)
        # 如果 coef > 0, 越多阴影 => cost 越低 => 越倾向选择阴影
        # 如果 coef < 0, 越多阴影 => cost 越高 => 越倾向选择太阳
        cost = edge_length * (1 - coef * shadow_ratio)
        shadow_lengths.append(cost)

    gdf_edges['cool_weight'] = shadow_lengths

    # 把新的权重写回图
    for (u, v, k), val in zip(gdf_edges.index, gdf_edges['cool_weight']):
        if (u, v, k) in G.edges:
            G[u][v][k]['cool_weight'] = val

    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        return None

    # 找最近节点
    orig_node = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest_node = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    # 最短路
    new_cool_route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='cool_weight')
    new_cool_route_edges = [(new_cool_route[i], new_cool_route[i+1], 0)
                            for i in range(len(new_cool_route)-1)]
    new_cool_route_gdf = gdf_edges.loc[new_cool_route_edges]

    # 将“Wanted”路线稍微平移，避免和“Shortest”完全重合
    new_cool_route_gdf = new_cool_route_gdf.copy()
    new_cool_route_gdf['geometry'] = new_cool_route_gdf.geometry.apply(
        lambda g: translate(g, xoff=1.5, yoff=1.5)
    )
    return new_cool_route_gdf

# -------------------------
# 原先的两个按钮：更新路线 & 生成路径
# -------------------------
initial_coef = 1
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
    'Weight is from -1 (ride under SUN) to 1 (ride under SHADOW).',
    ha='center',
    va='center',
    transform=ax_coef.transAxes
)

ax_button_update = plt.axes([0.8, 0.05, 0.1, 0.075])
button_update = Button(ax_button_update, 'Update Route')

def update_route(event):
    coef_val = coef_slider.val
    # 用当前 shadow_union 做计算
    new_route_gdf = update_cool_route(coef_val, shadow_union)

    # 清除之前的 Wanted Bike Route
    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    # 画新的 Wanted Bike Route
    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
    plt.legend(handles=[shortest_route_legend, wanted_route_legend],
               loc='upper right', bbox_to_anchor=(-2, 1.05))
    plt.draw()

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

    # 清除已有的 Shortest / Wanted 路线
    for artist in ax.lines + ax.collections:
        if artist.get_label() in ['Shortest Bike Route', 'Wanted Bike Route']:
            artist.remove()

    # 画最短路径
    route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
    route_edges = [(route[i], route[i+1], 0) for i in range(len(route)-1)]
    route_gdf = gdf_edges.loc[route_edges]
    route_gdf.plot(ax=ax, color='red', linewidth=2, label='Shortest Bike Route')

    # 画 Wanted Bike Route
    coef_val = coef_slider.val
    new_route_gdf = update_cool_route(coef_val, shadow_union)
    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')

    plt.legend(handles=[shortest_route_legend, wanted_route_legend],
               loc='upper right', bbox_to_anchor=(-2, 1.05))
    plt.draw()

button_generate.on_clicked(generate_path)

# 新增：清除起点和终点的按钮
ax_button_clear = plt.axes([0.8, 0.15, 0.1, 0.075])
button_clear = Button(ax_button_clear, 'Clear Points')
def clear_points(event):
    global click_count, origin_point_wgs84, destination_point_wgs84
    global origin_marker, destination_marker

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

    # 清除路线
    for artist in ax.lines + ax.collections:
        if artist.get_label() in ['Shortest Bike Route', 'Wanted Bike Route', 'Origin', 'Destination']:
            artist.remove()

    plt.draw()

button_clear.on_clicked(clear_points)

# --------------------------------------------------------
# 关键改动：添加“时间”滑块，每当时间变化，就重新计算阴影
# --------------------------------------------------------
ax_time = plt.axes([0.1, 0.05, 0.4, 0.03])
time_slider = Slider(
    ax=ax_time,
    label='Time (minutes from base_time)',
    valmin=0,
    valmax=120,       # 这里演示从起始时间往后 0~120 分钟
    valinit=0,
    valstep=5         # 每 5 分钟一个刻度
)

def on_time_slider_change(val):
    """
    当用户拖动时间滑块时，计算新的 current_time 并重新计算阴影 & 更新路线
    """
    global building_gdf_with_shadow, shadow_union

    # 计算新时间
    current_time = base_time + timedelta(minutes=val)
    print(f"当前时间: {current_time.isoformat()}")

    # 重新计算阴影
    new_bldg_gdf, new_shadow_union = recalc_sun_and_shadow(building_gdf, current_time)
    if (new_bldg_gdf is None) or (new_shadow_union is None):
        print("太阳在地平线以下或计算异常，阴影无效！")
        return

    building_gdf_with_shadow = new_bldg_gdf
    shadow_union = new_shadow_union

    # 在图上更新阴影显示
    # (1) 先把旧的 shadow 绘制对象去除
    for artist in ax.collections:
        # 注意: 这里要小心判断, 只移除 label='shadow' 的
        if artist.get_label() == 'shadow':
            artist.remove()

    # (2) 再绘制新的阴影
    shadow_gdf_new = building_gdf_with_shadow.dropna(subset=['shadow']).set_geometry('shadow')
    shadow_gdf_new.plot(ax=ax, color='gray', alpha=0.5, label='shadow')

    # 如果已经生成过路径，则可强制重新调用 generate_path 或 update_route
    # 这里为了演示方便，直接调用 generate_path(None)，
    # 它会先清掉旧路线，然后重新画一次。
    generate_path(None)

    plt.draw()

time_slider.on_changed(on_time_slider_change)

# 显示图形
plt.show()
