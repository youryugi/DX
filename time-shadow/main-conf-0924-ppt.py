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
# 1) Load precomputed (u,v,k, time) -> shadow_ratio
# Note: Please ensure consistency with G and time slicing below
# --------Update cost-------------------------------------------------------------------------------

# File paths for building and road GML files
bldg_directory = "bldg"
bldg_gml_files = [
    r"52350338_bldg_6697_op.gml",
    r"52350328_bldg_6697_op.gml",
    r"52350318_bldg_6697_op.gml",
    r"52350339_bldg_6697_op.gml",
    r"52350329_bldg_6697_op.gml",
    r"52350319_bldg_6697_op.gml",
    r"52350430_bldg_6697_op.gml",
    r"52350420_bldg_6697_op.gml",
    r"52350410_bldg_6697_op.gml"
]
road_directory = "tran"
road_gml_files = [
    r"52350338_tran_6697_op.gml",
    r"52350328_tran_6697_op.gml",
    r"52350318_tran_6697_op.gml",
    r"52350339_tran_6697_op.gml",
    r"52350329_tran_6697_op.gml",
    r"52350319_tran_6697_op.gml",
    r"52350430_tran_6697_op.gml",
    r"52350420_tran_6697_op.gml",
    r"52350410_tran_6697_op.gml"
]
# Path to save merged building data as .pkl
pkl_path = r"bldg_merged_LL_135.4748_34.6749_UR_135.5129_34.7026.pkl"
# Load precomputed edge shadow ratios
with open("edge_shadow_ratios_20250924_1400_1700_1min_LL_135.4748_34.6749_UR_135.5129_34.7026.pkl", "rb") as f:
    precomputed = pickle.load(f)
shadow_file = r"shadows_20250924_1400_1700_1min_LL_135.4748_34.6749_UR_135.5129_34.7026.pkl"
# Set manual or interactive mode for selecting points
manual_input_mode = False

manual_origin_point_wgs84 = (34.62709838787363, 135.5151808481631) # Large map origin
manual_destination_point_wgs84 = (34.64854640692838, 135.54677174184593) # Large map destination

# ---------------------------------------------------------------------------------------
# Intersection counter (remove if not needed)
# ---------------------------------------------------------------------------------------
intersection_counter = 0

# Legend handles (for plotting)
#shortest_route_legend = Line2D([0], [0], color='red', linewidth=2, label='Shortest Bike Route')
#wanted_route_legend   = Line2D([0], [0], color='green', linewidth=2, label='Wanted Bike Route (Dynamic A*)')
bigfontsize = 14

# Start timing for loading and processing
start_time = time.time()

if os.path.exists(pkl_path):
    print("Found existing .pkl cache, loading directly...")
    with open(pkl_path, 'rb') as f:
        bldg_merged_gdf = pickle.load(f)
else:
    print("No .pkl cache found, reading and merging GML files...")
    bldg_gdf_list = [gpd.read_file(f"{bldg_directory}{os.path.sep}{file}") for file in bldg_gml_files]
    bldg_merged_gdf = pd.concat(bldg_gdf_list, ignore_index=True)

    print("Saving as .pkl...")
    with open(pkl_path, 'wb') as f:
        pickle.dump(bldg_merged_gdf, f)

end_time = time.time()
print("Time taken for reading and processing:", end_time - start_time, "seconds")
print(bldg_merged_gdf.head())

# ---------------------------------------------------------------------------------------
# Read road GML files
# ---------------------------------------------------------------------------------------
starttime2 = time.time()

road_gdf_list = [gpd.read_file(f"{road_directory}{os.path.sep}{file}") for file in road_gml_files]
merged_road_gdf = pd.concat(road_gdf_list, ignore_index=True)
endtime2 = time.time()
readtrantime = endtime2 - starttime2
print("Time taken to read road data:", readtrantime)

building_gdf = bldg_merged_gdf
road_gdf = merged_road_gdf

# Ensure projection is EPSG:6669
if building_gdf.crs.to_epsg() != 6669:
    building_gdf = building_gdf.to_crs(epsg=6669)
if road_gdf.crs.to_epsg() != 6669:
    road_gdf = road_gdf.to_crs(epsg=6669)

# ---------------------------------------------------------------------------------------
# calculate_shadow_stats (for demonstration; otherwise, use precomputed)
# ---------------------------------------------------------------------------------------
def calculate_shadow_stats(route_gdf, time_to_union, start_time, sample_interval=60):
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
                # Previous intersection calculation, for reference
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

    print(f"Distance in shadow: {shadow_distance:.2f} m, "
          f"Distance in sunlight: {non_shadow_distance:.2f} m, "
          f"Total distance: {total_distance:.2f} m")
    print("Intersection calculations:", intersection_counter)
    return non_shadow_distance, shadow_distance

def set_shadesunlight_plot(sun_part, shade_part):
    """
    Makes stacked histogram to show proportions of shade/sunlit path
    """
    global proportions_ax

    proportions_ax.clear()

    columns = ("Shortest path", "Shade-aware path")
    proportions_ax.bar(columns, sun_part, 0.5, label="Sun-exposed distances", color='orange')
    proportions_ax.bar(columns, shade_part, 0.5, label="Shaded distances", bottom=sun_part, color='lightblue')
    proportions_ax.set_ylabel("Distance (m)")

    proportions_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),  # (x, y) 坐标，x=0.5为居中，y>1在图外上方
        ncol=2,
        fontsize=12
    )
    proportions_ax.set_title("Sun-exposed and shaded distances", fontsize=16, pad=30)

    proportions_ax.text(x=0, y=sun_part[0]/2, s=f"{sun_part[0]:.0f}", ha='center', va='center')
    proportions_ax.text(x=0, y=shade_part[0]/2 + sun_part[0], s=f"{shade_part[0]:.0f}", ha='center', va='center')

    proportions_ax.text(x=1, y=sun_part[1]/2, s=f"{sun_part[1]:.0f}", ha='center', va='center')
    proportions_ax.text(x=1, y=shade_part[1]/2 + sun_part[1], s=f"{shade_part[1]:.0f}", ha='center', va='center')

# ---------------------------------------------------------------------------------------
# Load time_to_union (shadow area for each time), and find nearest_time
# ---------------------------------------------------------------------------------------
with open(shadow_file, 'rb') as f:
    time_to_union = pickle.load(f)

def find_nearest_time(keys, target):
    return min(keys, key=lambda t: abs(t - target))

date_time = datetime(2025, 9, 24, 14, 30, tzinfo=timezone(timedelta(hours=9)))
nearest_time = find_nearest_time(time_to_union.keys(), date_time)
shadow_union = time_to_union[nearest_time]

if shadow_union is None:
    print("The sun is below the horizon or no valid shadow at this time.")
    exit()

shadow_gdf = gpd.GeoDataFrame(geometry=[shadow_union], crs='EPSG:6669')

# ---------------------------------------------------------------------------------------
# Prepare for plotting
# ---------------------------------------------------------------------------------------
#plt.rcParams['font.family'] = 'Lato'
fig, axs = plt.subplots(
    ncols=2,
    figsize=(16, 8),  # 可以适当加大
    gridspec_kw={'width_ratios': [1, 1]}  # 左侧是右侧的4倍宽
)

ax = axs[0]
proportions_ax = axs[1]

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
    Line2D([0], [0], color='green', linewidth=2, label='Shade-aware Route'),
]

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])

ax.set_title("Roads, buildings, and shadows", fontsize=16)
ax.legend(handles=legend_handles, ncol=2, loc='upper center', bbox_to_anchor=(0.0, -1.15, 1.0, 1.0), fontsize=bigfontsize)
#ax.set_xlabel("X (m)", fontsize=bigfontsize)
#ax.set_ylabel("Y (m)", fontsize=bigfontsize)
#ax.set_xticklabels(fontsize=bigfontsize)
#ax.set_yticklabels(fontsize=bigfontsize)
ax.grid(True)

set_shadesunlight_plot([0.0, 0.0], [0.0, 0.0])
shortest_path_sunlit_length = 0.0
shortest_path_shaded_length = 0.0

# ---------------------------------------------------------------------------------------
# Load OSMnx graph
# ---------------------------------------------------------------------------------------
building_gdf_wgs84 = building_gdf.to_crs(epsg=4326)
building_bounds_wgs84 = building_gdf_wgs84.total_bounds
bbox = (building_bounds_wgs84[3], building_bounds_wgs84[1],
        building_bounds_wgs84[2], building_bounds_wgs84[0])
map_id = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
osm_file = f"osmnx_graph_{map_id}.pkl"

if os.path.exists(osm_file):
    print(f"Loading local OSM data: {osm_file}")
    with open(osm_file, "rb") as f:
        G = pickle.load(f)
else:
    print("Downloading OSM data...")
    G = ox.graph_from_bbox(north=bbox[0], south=bbox[1], east=bbox[2], west=bbox[3], network_type="bike")
    print('Download complete')
    with open(osm_file, "wb") as f:
        pickle.dump(G, f)

print(f"Number of nodes: {len(G.nodes)}")
print(f"Number of edges: {len(G.edges)}")

# Extract edges as GeoDataFrame and project to EPSG:6669
gdf_edges = ox.graph_to_gdfs(G, nodes=False)
if gdf_edges.crs.to_epsg() != 4326:
    gdf_edges = gdf_edges.to_crs(epsg=4326)
gdf_edges = gdf_edges.to_crs(epsg=6669)

# Merge shadow polygons (for visualization only)
shadow_union = unary_union(shadow_gdf.geometry)

# ---------------------------------------------------------------------------------------
# Interactive point selection
# ---------------------------------------------------------------------------------------
transformer_to_wgs84 = Transformer.from_crs(6669, 4326, always_xy=True)
click_count = 0
origin_point_wgs84 = None
destination_point_wgs84 = None
origin_marker = None
destination_marker = None

proj_crs = 'EPSG:6669'
transformer_from_wgs84 = Transformer.from_crs('EPSG:4326', proj_crs, always_xy=True)

def on_map_click(event):
    """
    Routine used to capture the origin and destination points on the map.
    When the destination is captured, the route calculation is triggered immediately and displayed
    as soon as the results become available.
    Clicking again on the map will discard previously calculated routes and set a new origin point.
    """
    global click_count, origin_point_wgs84, destination_point_wgs84
    global origin_marker, destination_marker

    if manual_input_mode:
        if click_count == 0:
            proportions_ax.clear()
            origin_point_wgs84 = manual_origin_point_wgs84
            x, y = transformer_from_wgs84.transform(origin_point_wgs84[1], origin_point_wgs84[0])
            origin_marker = ax.plot(x, y, marker='o', color='blue', markersize=8, label='Origin')[0]
            plt.draw()
            click_count = 1
            print(f"Reproduced origin: (lat={origin_point_wgs84[0]}, lon={origin_point_wgs84[1]})")
        elif click_count == 1:
            destination_point_wgs84 = manual_destination_point_wgs84
            x, y = transformer_from_wgs84.transform(destination_point_wgs84[1], destination_point_wgs84[0])
            destination_marker = ax.plot(x, y, marker='o', color='magenta', markersize=8, label='Destination')[0]
            plt.draw()
            click_count = 0
            print(f"Reproduced destination: (lat={destination_point_wgs84[0]}, lon={destination_point_wgs84[1]})")
        return

    if event.inaxes != ax:
        return

    x_coord, y_coord = event.xdata, event.ydata
    lon, lat = transformer_to_wgs84.transform(x_coord, y_coord)

    if click_count == 0:
        clear_points()
        origin_point_wgs84 = (lat, lon)
        if origin_marker is not None:
            origin_marker.remove()
        if destination_marker is not None:
            destination_marker.remove()
        origin_marker = ax.plot(x_coord, y_coord, marker='o', color='blue', markersize=8, label='Origin')[0]
        plt.draw()
        click_count += 1
        print(f"Selected origin: (lat={lat}, lon={lon})")
    elif click_count == 1:
        destination_point_wgs84 = (lat, lon)
        # if destination_marker is not None:
        #     destination_marker.remove()
        destination_marker = ax.plot(x_coord, y_coord, marker='o', color='magenta', markersize=8, label='Destination')[0]
        plt.draw()
        #click_count += 1
        print(f"Selected destination: (lat={lat}, lon={lon})")

        # Now that origin and destination have been decided, compute the path and display it

        generate_path()
        # Reset state so that next click resets the origin point
        click_count = 0

fig.canvas.mpl_connect('button_press_event', on_map_click)
start_time = date_time

# ---------------------------------------------------------------------------------------
# Key: Use precomputed shadow ratios in multi-state Dijkstra, not intersection()
# ---------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. A* Search: Multi-state + time-dependent cost + full timing report
# ----------------------------------------------------------------------------
def update_cool_route(coef, start_time, sample_interval=300):
    """Return GeoDataFrame: optimal shade-aware route + print full timing report"""

    # --- Origin and destination ---
    # Use manually selected points if available
    if origin_point_wgs84 is not None and destination_point_wgs84 is not None:
        origin_node = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
        goal_node   = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])
    else:
        origin_node = ox.distance.nearest_nodes(G, X=manual_origin_point_wgs84[1],      Y=manual_origin_point_wgs84[0])
        goal_node   = ox.distance.nearest_nodes(G, X=manual_destination_point_wgs84[1], Y=manual_destination_point_wgs84[0])

    # =============== Global timer ===============
    t_global_start       = time.time()
    t_cost_accum         = 0.0   # time-dependent cost total
    t_heuristic_accum    = 0.0   # heuristic total
    t_neighbor_accum     = 0.0   # neighbor traversal total
    td_calls             = 0     # cost function call count
    h_calls              = 0     # heuristic call count
    neighbor_visits      = 0     # neighbor visit count

    # =============== Function definitions ===============
    speed = 10 * 1000 / 3600  # m/s
    goal_lat, goal_lon = G.nodes[goal_node]['y'], G.nodes[goal_node]['x']

    def heuristic(node, cur_time_s):
        nonlocal t_heuristic_accum, h_calls
        t0 = time.time(); h_calls += 1
        lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
        dist = ox.distance.euclidean(lat, lon, goal_lat, goal_lon)
        # Estimate time to reach straight-line distance (conservative)
        ratio = 0.5  # Assume half shade, can be replaced with more complex prediction
        h_val = dist * (1 + coef * ratio)
        t_heuristic_accum += (time.time() - t0)
        return h_val

    def time_dependent_cost(u, v, k, arrival_dt):
        """Integrate edge cost by sample_interval, return (cost, duration)"""
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
            cost_acc += length - coef * ratio * length
            tmp_dt  += timedelta(seconds=dt)
            remain  -= dt
        t_cost_accum += (time.time() - t0)
        return cost_acc, duration

    # =============== A* main loop ===============
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
        print("[A*] No feasible path found"); return None

    # =============== Backtrack latest timestamp state ===============
    goal_states = [(t, key) for key,t in [(k[0],k[1]) for k in pred if k[0]==goal_node]]
    best_t = max([t for t,_ in goal_states])
    nodes = [goal_node]; cur = (goal_node, best_t)
    while cur in pred:
        prev = pred[cur]; nodes.append(prev[0]); cur = prev
    nodes.reverse()

    # =============== Assemble GeoDataFrame ===============
    edges=[]
    for i in range(len(nodes)-1):
        u,v = nodes[i], nodes[i+1]
        k = 0 if (u,v,0) in gdf_edges.index else next((kk for kk in G[u][v] if (u,v,kk) in gdf_edges.index),0)
        edges.append((u,v,k))
    route = gdf_edges.loc[edges].copy()

    # =============== Print timing report ===============
    print("======= A★ Path‑finding Timing Report =======")
    print(f"Total elapsed              : {time.time()-t_global_start:.4f} s")
    print(f"Heuristic calls / time     : {h_calls} / {t_heuristic_accum:.4f} s")
    print(f"Cost calls      / time     : {td_calls} / {t_cost_accum:.4f} s")
    print(f"Neighbor visits  / time    : {neighbor_visits} / {t_neighbor_accum:.4f} s")
    print("=============================================\n")

    return route

# ---------------------------------------------------------------------------------------
# Set up Slider to set the importance of the shade in the route calculation
# ---------------------------------------------------------------------------------------
initial_coef = 0.5
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.subplots_adjust(left=0.1, bottom=0.3)

ax_coef = plt.axes([0.60, 0.1, 0.3, 0.03])  # Move slider to the right
coef_slider = Slider(
    ax=ax_coef,
    label='Shade weight',
    valmin=-1.0,
    valmax=1.0,
    valinit=initial_coef,
    valstep=0.1
)
ax_coef.text(
    0.5, -1.2,
    'Weight is from -1 (prefer sunlight) to 1 (prefer shade). ',
    ha='center',
    va='center',
    transform=ax_coef.transAxes
)

def update_route(event):
    """
    Routine used to recompute the shade-aware route when the value on the slider has been changed.
    """
    # First, check that the origin and destination are indeed set
    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        return

    coef_val = coef_slider.val
    new_route_gdf = update_cool_route(coef_val, start_time)

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    if new_route_gdf is not None:
        new_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')
    plt.draw()
    print("wanted route:")
    shade_aware_sunlit, shade_aware_shaded = calculate_shadow_stats(new_route_gdf, time_to_union, start_time)

    set_shadesunlight_plot([shortest_path_sunlit_length, shade_aware_sunlit], [shortest_path_shaded_length, shade_aware_shaded])


coef_slider.on_changed(update_route)

def generate_path():
    """
    Computes and displays on the figure the shortest route and the shade-aware route
    """
    if origin_point_wgs84 is None or destination_point_wgs84 is None:
        print("Please click on the map to select the start and end points first.")
        return

    orig_node = ox.distance.nearest_nodes(G, X=origin_point_wgs84[1], Y=origin_point_wgs84[0])
    dest_node = ox.distance.nearest_nodes(G, X=destination_point_wgs84[1], Y=destination_point_wgs84[0])

    starttimeshortpath = time.time()
    route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='length')
    endtimeshortpath = time.time()
    timeshortpath = endtimeshortpath - starttimeshortpath
    print("Time to calculate shortest path:", timeshortpath)

    route_edges = [(route[i], route[i+1], 0) for i in range(len(route) - 1)]
    route_gdf = gdf_edges.loc[route_edges]

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Shortest Bike Route':
            artist.remove()
    route_gdf.plot(ax=ax, color='red', linewidth=2, label='Shortest Bike Route')

    coef_val = coef_slider.val
    shade_aware_route_gdf = update_cool_route(coef_val, start_time)

    for artist in ax.lines + ax.collections:
        if artist.get_label() == 'Wanted Bike Route':
            artist.remove()

    print("Shortest route:")
    global shortest_path_shaded_length, shortest_path_sunlit_length
    shortest_path_sunlit_length, shortest_path_shaded_length = calculate_shadow_stats(route_gdf, time_to_union, start_time)

    print("Shade-aware route:")
    shade_aware_sunlit, shade_aware_shaded = calculate_shadow_stats(shade_aware_route_gdf, time_to_union, start_time)

    if shade_aware_route_gdf is not None:
        shade_aware_route_gdf.plot(ax=ax, color='green', linewidth=2, label='Wanted Bike Route')

    set_shadesunlight_plot([shortest_path_sunlit_length, shade_aware_sunlit], [shortest_path_shaded_length, shade_aware_shaded])

    plt.draw()
    print("wanted route:")

def clear_points():
    """
    Routine that will erase the path plotted on the graph.
    It is called when a new origin point is set, erasing any previous path
    """
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

# ---------------------------------------------------------------------------------------
# North Arrow
# ---------------------------------------------------------------------------------------
from matplotlib.offsetbox import AnchoredText
north_arrow = AnchoredText('↑ North', loc='upper left', pad=0, prop=dict(size=14), frameon=False)
ax.add_artist(north_arrow)

# 经纬度转投影坐标
x_here, y_here = transformer_from_wgs84.transform(135.486278, 34.689417)

# 在地图上画点
#ax.plot(x_here, y_here, marker='v', color='red', markersize=4, label='We are here')

# 添加带箭头的注释
# ax.annotate(
#     "we are here",
#     xy=(x_here, y_here+180),                # 点的位置
#     xytext=(x_here, y_here+50),      # 文字的位置（可调整150为你想要的距离）
#     color='red',
#     fontsize=12,
#     ha='center',
#     va='bottom',
#
# )

plt.show()
