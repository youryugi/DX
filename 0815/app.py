import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx

# ========= 配置 =========
GRAPHML_PATH = r"C:\Users\79152\Desktop\github\DX\0815\road_graph_with_accidents.graphml"
DEFAULT_ZOOM = 12
OSM_ATTR = "© OpenStreetMap contributors"
# =======================

st.set_page_config(page_title="安全路径规划 (OSM)", layout="wide")

@st.cache_resource(show_spinner=True)
def load_graphs(graphml_path: str):
    G_proj = ox.load_graphml(graphml_path)
    G_wgs = ox.project_graph(G_proj, to_crs="EPSG:4326")
    nodes = ox.graph_to_gdfs(G_wgs, edges=False)
    center_lat = float(nodes.geometry.y.mean())
    center_lon = float(nodes.geometry.x.mean())
    for _, _, _, data in G_proj.edges(keys=True, data=True):
        if "accident_count" not in data:
            data["accident_count"] = 0
        if "length" not in data:
            data["length"] = float(ox.distance.euclidean_dist_vec(
                G_proj.nodes[data["u"]]["y"], G_proj.nodes[data["u"]]["x"],
                G_proj.nodes[data["v"]]["y"], G_proj.nodes[data["v"]]["x"]
            ))
    return G_proj, G_wgs, (center_lat, center_lon)

G_proj, G_wgs, (center_lat, center_lon) = load_graphs(GRAPHML_PATH)

# ============== 侧边栏参数 ==============
with st.sidebar:
    st.header("参数")
    beta = st.slider("β（避险权重，越大越绕）", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    show_layers = st.multiselect(
        "底图图层（可多选切换）",
        ["OpenStreetMap", "CartoDB Positron", "CartoDB DarkMatter"],
        default=["OpenStreetMap"]
    )
    st.markdown("**操作提示**：点击“选择起点”或“选择终点”按钮后，在地图上点击设置对应点。")

# ============== Session 状态 ==============
if "start" not in st.session_state: st.session_state.start = None
if "end" not in st.session_state: st.session_state.end = None
if "last_clicked" not in st.session_state: st.session_state.last_clicked = None
if "select_mode" not in st.session_state: st.session_state.select_mode = None
if "map_center" not in st.session_state: st.session_state.map_center = (center_lat, center_lon)
if "map_zoom" not in st.session_state: st.session_state.map_zoom = DEFAULT_ZOOM
if "routes_geojson" not in st.session_state: st.session_state.routes_geojson = None

# ============== 主界面按钮 ==============
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if st.button("选择起点", type="secondary"):
        st.session_state.select_mode = "start"
with col2:
    if st.button("选择终点", type="secondary"):
        st.session_state.select_mode = "end"
with col3:
    if st.button("重置起终点", type="secondary"):
        st.session_state.start = None
        st.session_state.end = None
        st.session_state.routes_geojson = None
        st.session_state.select_mode = None
with col4:
    compute_btn = st.button("计算路径（最短 + 避险）", type="primary")

export_btn = st.button("导出路径", type="secondary")

# ============== 鼠标样式切换 ==============
if st.session_state.select_mode in ["start", "end"]:
    st.markdown("""
        <style>
        .folium-map { cursor: crosshair !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .folium-map { cursor: grab !important; }
        </style>
        """, unsafe_allow_html=True)

# ============== 地图渲染 ==============
m = folium.Map(
    location=[st.session_state.map_center[0], st.session_state.map_center[1]],
    zoom_start=st.session_state.map_zoom,
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr=OSM_ATTR
)

tile_defs = {
    "CartoDB Positron": {
        "tiles": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": "© OpenStreetMap contributors, © CARTO",
    },
    "CartoDB DarkMatter": {
        "tiles": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attr": "© OpenStreetMap contributors, © CARTO",
    },
}
for name in show_layers:
    if name != "OpenStreetMap" and name in tile_defs:
        folium.TileLayer(tiles=tile_defs[name]["tiles"], name=name, attr=tile_defs[name]["attr"]).add_to(m)

# 起点终点 marker
if st.session_state.start:
    folium.Marker(st.session_state.start, icon=folium.Icon(color="green"), tooltip="起点").add_to(m)
if st.session_state.end:
    folium.Marker(st.session_state.end, icon=folium.Icon(color="blue"), tooltip="终点").add_to(m)

# 路径渲染
if st.session_state.routes_geojson:
    routes_geojson = st.session_state.routes_geojson
    coords_short = [(lat, lon) for lon, lat in routes_geojson["short"]["geometry"]["coordinates"]]
    coords_safe = [(lat, lon) for lon, lat in routes_geojson["safe"]["geometry"]["coordinates"]]
    folium.PolyLine(coords_short, color="blue", weight=6, opacity=0.8,
                    tooltip=f"最短路径 | 长度 {routes_geojson['short']['properties']['length_m']/1000:.2f} km | 事故累计 {routes_geojson['short']['properties']['accidents_sum']}").add_to(m)
    folium.PolyLine(coords_safe, color="red", weight=6, opacity=0.9,
                    tooltip=f"避险路径(β={beta}) | 长度 {routes_geojson['safe']['properties']['length_m']/1000:.2f} km | 事故累计 {routes_geojson['safe']['properties']['accidents_sum']}").add_to(m)

map_ret = st_folium(
    m,
    height=700,
    use_container_width=True,
    returned_objects=["last_clicked", "center", "zoom"],
    key="picker_map"
)

# ⚠️ 只有在用户真的操作时才更新，不要每次都重置
if map_ret:
    if map_ret.get("center"):
        c = map_ret["center"]
        if isinstance(c, dict) and "lat" in c and "lng" in c:
            st.session_state.map_center = (c["lat"], c["lng"])
    if map_ret.get("zoom") is not None:
        st.session_state.map_zoom = map_ret["zoom"]

    # 处理点击选点
    if map_ret.get("last_clicked") is not None:
        lat = map_ret["last_clicked"]["lat"]
        lng = map_ret["last_clicked"]["lng"]
        if st.session_state.last_clicked != (lat, lng):
            if st.session_state.select_mode == "start":
                st.session_state.start = (lat, lng)
                st.toast("已设置起点 ✅", icon="✅")
                st.session_state.select_mode = None
            elif st.session_state.select_mode == "end":
                st.session_state.end = (lat, lng)
                st.toast("已设置终点 ✅", icon="✅")
                st.session_state.select_mode = None
            st.session_state.last_clicked = (lat, lng)

# ============== 路径计算函数 ==============
def find_route_nodes(G_wgs, G_proj, start, end, beta):
    start_node = ox.nearest_nodes(G_wgs, X=start[1], Y=start[0])
    end_node = ox.nearest_nodes(G_wgs, X=end[1], Y=end[0])
    path_short = nx.shortest_path(G_proj, source=start_node, target=end_node, weight="length")
    def safe_weight(u, v, k, data):
        return data.get("length", 1) + beta * data.get("accident_count", 0) * 10
    path_safe = nx.shortest_path(G_proj, source=start_node, target=end_node, weight=safe_weight)
    return path_short, path_safe

def path_stats(G_proj, path):
    length_sum = 0
    accident_sum = 0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G_proj.get_edge_data(u, v)
        if isinstance(edge_data, dict):
            data = list(edge_data.values())[0] if len(edge_data) > 0 else {}
        else:
            data = edge_data if edge_data else {}
        length_sum += data.get("length", 0)
        accident_sum += data.get("accident_count", 0)
    return length_sum, accident_sum

def path_to_coords(G_wgs, path):
    coords = []
    for node in path:
        node_data = G_wgs.nodes[node]
        coords.append((node_data['y'], node_data['x']))
    return coords

# ============== 路径计算 ==============
if compute_btn:
    if st.session_state.start and st.session_state.end:
        with st.spinner("正在计算路径…"):
            path_short, path_safe = find_route_nodes(G_wgs, G_proj, st.session_state.start, st.session_state.end, beta)
        len_short, acc_short = path_stats(G_proj, path_short)
        len_safe,  acc_safe  = path_stats(G_proj, path_safe)
        coords_short = path_to_coords(G_wgs, path_short)
        coords_safe = path_to_coords(G_wgs, path_safe)
        st.session_state.routes_geojson = {
            "short": {
                "type": "Feature",
                "properties": {"name": "shortest", "length_m": len_short, "accidents_sum": acc_short},
                "geometry": {"type": "LineString", "coordinates": [(lon, lat) for lat, lon in coords_short]},
            },
            "safe": {
                "type": "Feature",
                "properties": {"name": "safe", "length_m": len_safe, "accidents_sum": acc_safe, "beta": beta},
                "geometry": {"type": "LineString", "coordinates": [(lon, lat) for lat, lon in coords_safe]},
            }
        }
    else:
        st.warning("请先在地图上点击设置**起点**和**终点**。")

# ============== 路径导出 ==============
if export_btn:
    routes_geojson = st.session_state.get("routes_geojson")
    if not routes_geojson:
        st.warning("请先计算路径。")
    else:
        import json, time
        fname = f"routes_{int(time.time())}.geojson"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": [routes_geojson["short"], routes_geojson["safe"]]}, f, ensure_ascii=False)
        st.success(f"已导出：{fname}")
        st.download_button("下载 GeoJSON", data=json.dumps({"type":"FeatureCollection","features":[routes_geojson["short"],routes_geojson["safe"]]}, ensure_ascii=False).encode("utf-8"),
                           file_name=fname, mime="application/geo+json")
