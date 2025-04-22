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
# 在代码开头处定义图例句柄
shortest_route_legend = Line2D([0], [0], color='red', linewidth=2, label='Shortest Bike Route')
wanted_route_legend = Line2D([0], [0], color='green', linewidth=2, label='Wanted Bike Route')


bldg_gml_files = [
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357451_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357452_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357453_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357461_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357462_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357463_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357471_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357472_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357473_bldg_6697_op.gml"

]

# 读取每个 GML 文件并存储到列表中
bldg_gdf_list = [gpd.read_file(file) for file in bldg_gml_files]

# 合并所有 GeoDataFrame
bldg_merged_gdf = pd.concat(bldg_gdf_list, ignore_index=True)
# -------------------------
road_gml_files = [
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357451_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357452_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357453_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357461_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357462_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357463_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357471_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357472_tran_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\tran\51357473_tran_6697_op.gml"
    # 以及其它所有 road gml
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
# 计算太阳高度角和方位角
# -------------------------
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo", latitude=34.6937, longitude=135.5023)
date_time = datetime(2024, 12, 5, 13, 10, tzinfo=timezone(timedelta(hours=9)))

solar_elevation = elevation(city.observer, date_time)
solar_azimuth = azimuth(city.observer, date_time)

print(f"太阳高度角: {solar_elevation:.2f}°")
print(f"太阳方位角: {solar_azimuth:.2f}°")

if solar_elevation <= 0:
    print("太阳位于地平线以下，无法生成阴影。")
    exit()

sun_vector = np.array([
    np.cos(np.radians(solar_elevation)) * np.sin(np.radians(solar_azimuth)),
    np.cos(np.radians(solar_elevation)) * np.cos(np.radians(solar_azimuth)),
    np.sin(np.radians(solar_elevation))
])

# 查找高度列
height_column = None
for col in building_gdf.columns:
    if 'height' in col.lower():
        height_column = col
        break

if height_column is not None:
    building_gdf[height_column] = building_gdf[height_column].fillna(3)
else:
    print("没有找到高度列，将默认高度设为3米用于计算阴影。")
    height_column = 'default_height'
    building_gdf[height_column] = 3.0

# -------------------------
# 阴影计算函数
# -------------------------
def shadow_using_lines(geometry, height):
    if sun_vector[2] <= 0:
        return None
    polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
    shadow_lines = []

    for poly in polygons:
        base_coords = [(x, y) for x, y, *rest in poly.exterior.coords]
        shadow_coords = [
            (
                x - height / np.tan(np.radians(solar_elevation)) * sun_vector[0],
                y - height / np.tan(np.radians(solar_elevation)) * sun_vector[1]
            )
            for x, y in base_coords
        ]

        for base, shadow in zip(base_coords, shadow_coords):
            shadow_lines.append(LineString([base, shadow]))

    union_lines = unary_union(shadow_lines)
    shadow_polygon = union_lines.convex_hull
    return shadow_polygon

# -------------------------
# 为每个建筑物生成阴影
# -------------------------
building_gdf['shadow'] = building_gdf.apply(
    lambda row: shadow_using_lines(row.geometry, row[height_column]), axis=1
)

if 'shadow' not in building_gdf.columns or building_gdf['shadow'].isnull().all():
    print("未生成任何有效阴影。请检查数据或逻辑。")
    exit()

shadow_gdf = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')

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
    Patch(facecolor='gray', edgecolor='black', label='Shadow'),
]
plt.title("Roads, buildings, and shadows     ↑North", fontsize=16)
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

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

def on_map_click(event):
    global click_count, origin_point_wgs84, destination_point_wgs84, origin_marker, destination_marker

    if event.inaxes != ax:
        return

    x_coord, y_coord = event.xdata, event.ydata
    lon, lat = transformer_to_wgs84.transform(x_coord, y_coord)

    if click_count == 0:
        # 第一次点击，设置起点
        origin_point_wgs84 = (lat, lon)
        if origin_marker is not None:
            origin_marker.remove()
        origin_marker = ax.plot(x_coord, y_coord, marker='o', color='blue', markersize=10, label='Origin')[0]
        plt.draw()
        click_count += 1
        print(f"已选择起点: (lat={lat}, lon={lon})")
    elif click_count == 1:
        # 第二次点击，设置终点
        destination_point_wgs84 = (lat, lon)
        if destination_marker is not None:
            destination_marker.remove()
        destination_marker = ax.plot(x_coord, y_coord, marker='o', color='magenta', markersize=10, label='Destination')[0]
        plt.draw()
        click_count += 1
        print(f"已选择终点: (lat={lat}, lon={lon})")

fig.canvas.mpl_connect('button_press_event', on_map_click)

def update_cool_route(coef):
    shadow_lengths = []
    for idx, row in gdf_edges.iterrows():
        edge_geom = row.geometry
        edge_length = edge_geom.length
        intersection_geom = edge_geom.intersection(shadow_union)
        shadowed_length = intersection_geom.length if not intersection_geom.is_empty else 0
        shadow_ratio = shadowed_length / edge_length if edge_length > 0 else 0

        cost = edge_length * (1 - coef * shadow_ratio)
        shadow_lengths.append(cost)

    gdf_edges['cool_weight'] = shadow_lengths

    for (u,v,k), val in zip(gdf_edges.index, gdf_edges['cool_weight']):
        if (u,v,k) in G.edges:
            G[u][v][k]['cool_weight'] = val

    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        return None

    orig_node = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest_node = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    new_cool_route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='cool_weight')
    new_cool_route_edges = [(new_cool_route[i], new_cool_route[i+1], 0) for i in range(len(new_cool_route)-1)]
    new_cool_route_gdf = gdf_edges.loc[new_cool_route_edges]

    new_cool_route_gdf = new_cool_route_gdf.copy()
    new_cool_route_gdf['geometry'] = new_cool_route_gdf.geometry.apply(lambda g: translate(g, xoff=1.5, yoff=1.5))
    return new_cool_route_gdf

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
    new_route_gdf = update_cool_route(coef_val)

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
    # 在绘制完成后手动更新图例
    plt.legend(handles=[shortest_route_legend, wanted_route_legend,], loc='upper right',          # 图例框锚点在图例的左上角
    bbox_to_anchor=(-2, 1.05))
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

    # 最短路径
    route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
    route_edges = [(route[i], route[i+1], 0) for i in range(len(route)-1)]
    route_gdf = gdf_edges.loc[route_edges]

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Shortest Bike Route':
            artist.remove()

    route_gdf.plot(ax=ax, color='red', linewidth=2, label='Shortest Bike Route')

    # Wanted Bike Route
    coef_val = coef_slider.val
    new_route_gdf = update_cool_route(coef_val)

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
    plt.legend(handles=[shortest_route_legend, wanted_route_legend,], loc='upper right',          # 图例框锚点在图例的左上角
    bbox_to_anchor=(-2, 1.05))
    plt.draw()

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

plt.show()
