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
from matplotlib.widgets import Slider, Button, TextBox
from shapely.affinity import translate

# -------------------------
# 数据加载与坐标处理
# -------------------------
building_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\bldg\51357451_bldg_6697_op.gml"
road_gml_file = r"C:\Users\79152\Downloads\27100_osaka-shi_city_2022_citygml_3_op\udx\tran\51357451_tran_6697_op.gml"

building_gdf = gpd.read_file(building_gml_file)
road_gdf = gpd.read_file(road_gml_file)

# 确保建筑物和道路使用 EPSG:6669
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

if height_column is None:
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

# 计算初始阴影
building_gdf['shadow'] = building_gdf.apply(
    lambda row: shadow_using_lines(row.geometry, row[height_column]), axis=1
)
shadow_gdf = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')
shadow_union = unary_union(shadow_gdf.geometry)

# -------------------------
# 获取路网和路线规划
# -------------------------
building_gdf_wgs84 = building_gdf.to_crs(epsg=4326)
building_bounds_wgs84 = building_gdf_wgs84.total_bounds
bbox = (building_bounds_wgs84[3], building_bounds_wgs84[1], building_bounds_wgs84[2], building_bounds_wgs84[0])
G = ox.graph_from_bbox(north=bbox[0], south=bbox[1], east=bbox[2], west=bbox[3], network_type="bike")
gdf_edges = ox.graph_to_gdfs(G, nodes=False).to_crs(epsg=6669)

origin_point = (34.6266, 135.5133)
destination_point = (34.6390, 135.5190)
orig_node = ox.distance.nearest_nodes(G, X=origin_point[1], Y=origin_point[0])
dest_node = ox.distance.nearest_nodes(G, X=destination_point[1], Y=destination_point[0])

# -------------------------
# 阴凉路径函数
# -------------------------
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
    for (u, v, k), val in zip(gdf_edges.index, gdf_edges['cool_weight']):
        if (u, v, k) in G.edges:
            G[u][v][k]['cool_weight'] = val
    new_route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='cool_weight')
    new_route_edges = [(new_route[i], new_route[i+1], 0) for i in range(len(new_route)-1)]
    new_route_gdf = gdf_edges.loc[new_route_edges]
    new_route_gdf = new_route_gdf.copy()
    new_route_gdf['geometry'] = new_route_gdf.geometry.apply(lambda g: translate(g, xoff=1.5, yoff=1.5))
    return new_route_gdf

# 初始路径和阴凉路径
initial_coef = 1.0
cool_route_gdf = update_cool_route(initial_coef)

# -------------------------
# 绘图和交互
# -------------------------
plt.subplots_adjust(left=0.1, bottom=0.35)
fig, ax = plt.subplots(figsize=(12, 8))
road_gdf.plot(ax=ax, color='yellow', alpha=0.5, label='Road')
shadow_gdf.plot(ax=ax, color='gray', alpha=0.5, label='Shadow')
building_gdf.plot(ax=ax, color='lightblue', label='Building')
cool_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Coolest Bike Route')
plt.legend()

# 滑块
ax_coef = plt.axes([0.1, 0.2, 0.65, 0.03])
coef_slider = Slider(ax=ax_coef, label='Shadow Weight', valmin=0.0, valmax=1.0, valinit=initial_coef, valstep=0.1)

# 日期输入和按钮
ax_date_input = plt.axes([0.1, 0.05, 0.3, 0.05])
date_textbox = TextBox(ax_date_input, "Date (YYYY-MM-DD HH:MM): ", initial=date_time.strftime('%Y-%m-%d %H:%M'))
ax_now_button = plt.axes([0.42, 0.05, 0.1, 0.05])
now_button = Button(ax_now_button, "Now")
ax_update_button = plt.axes([0.55, 0.05, 0.1, 0.05])
update_button = Button(ax_update_button, "Update Shadow")

# 功能函数
def set_current_time(event):
    now = datetime.now(tz=timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')
    date_textbox.set_val(now)

def update_shadow_and_route(event):
    global solar_elevation, solar_azimuth, sun_vector, shadow_union
    try:
        input_date = datetime.strptime(date_textbox.text.strip(), '%Y-%m-%d %H:%M').replace(tzinfo=timezone(timedelta(hours=9)))
        solar_elevation = elevation(city.observer, input_date)
        solar_azimuth = azimuth(city.observer, input_date)
        sun_vector = np.array([
            np.cos(np.radians(solar_elevation)) * np.sin(np.radians(solar_azimuth)),
            np.cos(np.radians(solar_elevation)) * np.cos(np.radians(solar_azimuth)),
            np.sin(np.radians(solar_elevation))
        ])
        building_gdf['shadow'] = building_gdf.apply(lambda row: shadow_using_lines(row.geometry, row[height_column]), axis=1)
        shadow_gdf_updated = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')
        shadow_union = unary_union(shadow_gdf_updated.geometry)
        for artist in ax.collections:
            artist.remove()
        road_gdf.plot(ax=ax, color='yellow', alpha=0.5, label='Road')
        shadow_gdf_updated.plot(ax=ax, color='gray', alpha=0.5, label='Shadow')
        building_gdf.plot(ax=ax, color='lightblue', label='Building')
        new_cool_route_gdf = update_cool_route(coef_slider.val)
        new_cool_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Coolest Bike Route')
        plt.legend()
        plt.draw()
    except Exception as e:
        print(f"Error updating shadow and route: {e}")

now_button.on_clicked(set_current_time)
update_button.on_clicked(update_shadow_and_route)

plt.show()
