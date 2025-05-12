import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
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
bigfontsize=14
# 图例句柄
shortest_route_legend = Line2D([0], [0], color='red',   linewidth=2, label='Shortest Bike Route')
wanted_route_legend  = Line2D([0], [0], color='green', linewidth=2, label='Wanted Bike Route')

# -------------------------
# 数据加载与坐标处理
# -------------------------
building_gml_file = r"C:\Users\79152\Downloads\51357451_bldg_6697_op.gml"
road_gml_file     = r"C:\Users\79152\Downloads\51357451_tran_6697_op.gml"

building_gdf = gpd.read_file(building_gml_file)
road_gdf     = gpd.read_file(road_gml_file)

if building_gdf.crs.to_epsg() != 6669:
    building_gdf = building_gdf.to_crs(epsg=6669)
if road_gdf.crs.to_epsg() != 6669:
    road_gdf = road_gdf.to_crs(epsg=6669)

# -------------------------
# 太阳高度角与方位角
# -------------------------
city = LocationInfo(name="Osaka", region="Japan", timezone="Asia/Tokyo",
                    latitude=34.6937, longitude=135.5023)
date_time = datetime(2024, 12, 5, 13, 10,
                     tzinfo=timezone(timedelta(hours=9)))
date_time = datetime(2024, 9, 5, 15, 00,
                     tzinfo=timezone(timedelta(hours=9)))
solar_elevation = elevation(city.observer, date_time)
solar_azimuth   = azimuth(city.observer,  date_time)
print(f"太陽高度角: {solar_elevation:.2f}°")
print(f"太陽方位角: {solar_azimuth:.2f}°")

if solar_elevation <= 0:
    print("太陽は地平線の下にあり、影を作ることができない")
    exit()

sun_vector = np.array([
    np.cos(np.radians(solar_elevation)) * np.sin(np.radians(solar_azimuth)),
    np.cos(np.radians(solar_elevation)) * np.cos(np.radians(solar_azimuth)),
    np.sin(np.radians(solar_elevation))
])

# -------------------------
# 建筑物高度列
# -------------------------
height_column = next((c for c in building_gdf.columns if 'height' in c.lower()), None)
if height_column is not None:
    building_gdf[height_column] = building_gdf[height_column].fillna(3)
else:
    height_column = 'default_height'
    building_gdf[height_column] = 3.0
    print("高度の列が見つからなかったため、デフォルトの高さを3メートルとして影を計算します。")

# -------------------------
# 阴影计算
# -------------------------
def shadow_using_lines(geometry, height):
    if sun_vector[2] <= 0:
        return None
    polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
    shadow_lines = []
    for poly in polygons:
        base_coords = [(x, y) for x, y, *_ in poly.exterior.coords]
        shadow_coords = [
            (
                x - height / np.tan(np.radians(solar_elevation)) * sun_vector[0],
                y - height / np.tan(np.radians(solar_elevation)) * sun_vector[1]
            ) for x, y in base_coords
        ]
        for base, shad in zip(base_coords, shadow_coords):
            shadow_lines.append(LineString([base, shad]))
    return unary_union(shadow_lines).convex_hull

building_gdf['shadow'] = building_gdf.apply(
    lambda row: shadow_using_lines(row.geometry, row[height_column]), axis=1
)

if building_gdf['shadow'].isnull().all():
    print("有効な影が生成されませんでした。データまたはロジックを確認してください。")
    exit()

shadow_gdf  = building_gdf.dropna(subset=['shadow']).set_geometry('shadow')
shadow_union = unary_union(shadow_gdf.geometry)

# -------------------------
# 绘图
# -------------------------
plt.rcParams['font.family'] = 'SimHei'
fig, ax = plt.subplots(figsize=(12, 8))

bounds = building_gdf.total_bounds
buffer = 10
x_min, y_min, x_max, y_max = bounds - [buffer, buffer, -buffer, -buffer]
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

road_gdf.plot(ax=ax,     color='yellow',   alpha=0.5, label='road')
shadow_gdf.plot(ax=ax,   color='gray',     alpha=0.5, label='shadow')
building_gdf.plot(ax=ax, color='lightblue',            label='building')

legend_handles = [
    Patch(facecolor='yellow', label='Road'),
    Patch(facecolor='lightblue', label='Building'),
    Patch(facecolor='gray', edgecolor='gray', label='Shadow'),
    Line2D([0], [0], marker='o', color='blue', markersize=8, linestyle='None', label='Start point'),
    Line2D([0], [0], marker='o', color='magenta', markersize=8, linestyle='None', label='End point'),
    Line2D([0], [0], color='red', linewidth=2, label='Shortest route'),
    Line2D([0], [0], color='green', linewidth=2, label='Shade-aware route'),
]
plt.title("Roads, buildings, and shadows     ↑North\n大阪府　大阪市　阿倍野区　桃ケ池町１丁目", fontsize=16)
plt.legend(handles=legend_handles, loc='lower left',fontsize=bigfontsize)
plt.xlabel("X (m)", fontsize=bigfontsize)
plt.ylabel("Y (m)", fontsize=bigfontsize)
plt.xticks(fontsize=bigfontsize)
plt.yticks(fontsize=bigfontsize)

# -------------------------
# 路网
# -------------------------
building_gdf_wgs84   = building_gdf.to_crs(epsg=4326)
minx, miny, maxx, maxy = building_gdf_wgs84.total_bounds
bbox = (maxy, miny, maxx, minx)          # N, S, E, W

print('downloading OSM data...')
G = ox.graph_from_bbox(*bbox, network_type="drive")
print(f"number of nodes: {len(G.nodes)}")
print(f"number of edges: {len(G.edges)}")

gdf_edges = ox.graph_to_gdfs(G, nodes=False).to_crs(epsg=6669)

# -------------------------
# 交互
# -------------------------
transformer_to_wgs84 = Transformer.from_crs(6669, 4326, always_xy=True)
click_count = 0
origin_point_wgs84 = destination_point_wgs84 = None
origin_marker = destination_marker = None

def on_map_click(event):
    global click_count, origin_point_wgs84, destination_point_wgs84
    global origin_marker, destination_marker
    if event.inaxes != ax: return
    x_coord, y_coord = event.xdata, event.ydata
    lon, lat = transformer_to_wgs84.transform(x_coord, y_coord)
    if click_count == 0:                         # 起点
        origin_point_wgs84 = (lat, lon)
        if origin_marker: origin_marker.remove()
        origin_marker = ax.plot(x_coord, y_coord, 'ob', markersize=8, label='Origin')[0]
        click_count += 1
        print(f"start: (lat={lat}, lon={lon})")
    elif click_count == 1:                       # 终点
        destination_point_wgs84 = (lat, lon)
        if destination_marker: destination_marker.remove()
        destination_marker = ax.plot(x_coord, y_coord, 'om', markersize=8, label='Destination')[0]
        click_count += 1
        print(f"end: (lat={lat}, lon={lon})")
    plt.draw()

fig.canvas.mpl_connect('button_press_event', on_map_click)

# -------------------------
# 路线加权函数
# -------------------------
def update_cool_route(coef):
    costall = []
    for idx, row in gdf_edges.iterrows():
        edge_geom = row.geometry
        edge_length = edge_geom.length
        intersection_geom = edge_geom.intersection(shadow_union)
        shadowed_length = intersection_geom.length if not intersection_geom.is_empty else 0
        #sunny_dist = edge_length-shadowed_length
        cost = edge_length - coef * shadowed_length
        costall.append(cost)

    gdf_edges['cool_weight'] = costall

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


    # —— ② 仅用于绘图的平移副本 (不影响 cool_route_gdf) ——
    plot_gdf = new_cool_route_gdf.copy()
    plot_gdf['geometry'] = plot_gdf.geometry.apply(
        lambda g: translate(g, xoff=1.5, yoff=1.5)
    )
    return new_cool_route_gdf, plot_gdf

### === 新增：计算阴影/非阴影长度 === ###
def calc_shadow_stats(route_gdf):
    shadow_len = 0
    total_len  = 0
    for geom in route_gdf.geometry:
        total   = geom.length
        shaded  = geom.intersection(shadow_union).length
        shadow_len += shaded
        total_len  += total
    sunny_len = total_len - shadow_len
    return total_len, shadow_len, sunny_len
### =================================== ###

# -------------------------
# UI 控件
# -------------------------
initial_coef = 1
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.subplots_adjust(left=0.1, bottom=0.3)

ax_coef = plt.axes([0.1, 0.1, 0.65, 0.03])
coef_slider = Slider(ax=ax_coef, label='Shadow weight',
                     valmin=-1.0, valmax=1.0, valinit=initial_coef, valstep=0.1)
ax_coef.text(0.5, -1.2, 'Weight is from -1 (SUN) to 1 (SHADOW).',
             ha='center', va='center', transform=ax_coef.transAxes)

ax_button_update = plt.axes([0.8, 0.05, 0.1, 0.075])
button_update = Button(ax_button_update, 'Update Route')

def update_route(event):
    coef_val = coef_slider.val
    new_route_gdf,plot_gdf = update_cool_route(coef_val)

    # 移除旧线
    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    if new_route_gdf is not None:
        plot_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
        ### === 新增：统计打印 === ###
        tot, shad, sun = calc_shadow_stats(new_route_gdf)
        print(f"[Wanted Route] 总长: {tot:.1f} m, 阴影: {shad:.1f} m, 阳光: {sun:.1f} m")
        ### ======================= ###
    plt.legend(handles=[shortest_route_legend, wanted_route_legend], loc='upper right',
               bbox_to_anchor=(-2, 1.05))
    plt.draw()

button_update.on_clicked(update_route)

ax_button_generate = plt.axes([0.65, 0.15, 0.1, 0.075])
button_generate = Button(ax_button_generate, 'Generate Path')
ax_button_generate.text(-4.5, 0.65,
    'Please click on the map to select the start and end points',
    ha='center', va='center', transform=ax_button_generate.transAxes)

def generate_path(event):
    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        print("Please select start and destination first.")
        return

    orig = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    # 最短路径
    route      = nx.shortest_path(G, orig, dest, weight='length')
    route_edges = [(route[i], route[i+1], 0) for i in range(len(route)-1)]
    route_gdf   = gdf_edges.loc[route_edges]

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Shortest Bike Route':
            artist.remove()
    route_gdf.plot(ax=ax, color='red', linewidth=2, label='Shortest Bike Route')
    ### === 新增：统计打印 === ###
    tot, shad, sun = calc_shadow_stats(route_gdf)
    print(f"[Shortest Route] 总长: {tot:.1f} m, 阴影: {shad:.1f} m, 阳光: {sun:.1f} m")
    ### ======================= ###

    # Wanted Route
    coef_val = coef_slider.val
    new_route_gdf,plot_gdf = update_cool_route(coef_val)
    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()
    if new_route_gdf is not None:
        plot_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
        ### === 新增：统计打印 === ###
        tot, shad, sun = calc_shadow_stats(new_route_gdf)
        print(f"[Wanted Route] 总长: {tot:.1f} m, 阴影: {shad:.1f} m, 阳光: {sun:.1f} m")
        ### ======================= ###

    plt.legend(handles=[shortest_route_legend, wanted_route_legend], loc='upper right',
               bbox_to_anchor=(-2, 1.05))
    plt.draw()

button_generate.on_clicked(generate_path)

# 清除按钮
ax_button_clear = plt.axes([0.8, 0.15, 0.1, 0.075])
button_clear = Button(ax_button_clear, 'Clear Points')
def clear_points(event):
    global click_count, origin_point_wgs84, destination_point_wgs84
    global origin_marker, destination_marker
    click_count = 0
    origin_point_wgs84 = destination_point_wgs84 = None
    if origin_marker:      origin_marker.remove();      origin_marker = None
    if destination_marker: destination_marker.remove(); destination_marker = None
    for artist in ax.lines + ax.collections:
        if artist.get_label() in ['Shortest Bike Route', 'Wanted Bike Route',
                                  'Origin', 'Destination']:
            artist.remove()
    plt.draw()
button_clear.on_clicked(clear_points)
from matplotlib.offsetbox import AnchoredText

north_arrow = AnchoredText('↑ North', loc='upper left', pad=0, prop=dict(size=14), frameon=False)
ax.add_artist(north_arrow)
plt.show()
