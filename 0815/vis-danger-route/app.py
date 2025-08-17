from flask import Flask, render_template, request, jsonify
import networkx as nx
import osmnx as ox

app = Flask(__name__)

# 预加载路网（只加载一次）
GRAPHML_PATH = r"templates/road_graph_with_accidents.graphml"
G_proj = ox.load_graphml(GRAPHML_PATH)
G_wgs  = ox.project_graph(G_proj, to_crs="EPSG:4326")
print("jiazai wancheng ")
# 简单兜底：缺属性就补
for u, v, k, data in G_proj.edges(keys=True, data=True):
    data.setdefault("accident_count", 0)
    if "length" not in data:
        u_node, v_node = G_proj.nodes[u], G_proj.nodes[v]
        data["length"] = float(ox.distance.euclidean_dist_vec(u_node["y"], u_node["x"], v_node["y"], v_node["x"]))
# === 事故热力图数据：从现有 HTML 中提取点位 ===
import os, re, json
from flask import jsonify  # 顶部原来就有的话可忽略

HEATMAP_HTML_PATH = os.path.join(os.path.dirname(__file__), "accident_heatmap_osm.html")

def load_heat_points_from_html(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        # 抓取 L.heatLayer([...]) 里的数组
        m = re.search(r"L\.heatLayer\(\s*(\[\s*(?:\[[^\]]+\]\s*,?\s*)+\])", html, re.S)
        if not m:
            print("未在热力图 HTML 中找到 heat 点位数组")
            return []
        arr = m.group(1)
        pts = json.loads(arr)  # HTML 里的 [[lat,lng], ...] 本身就是合法 JSON
        return pts
    except Exception as e:
        print("加载事故热力点失败：", e)
        return []

ACCIDENT_POINTS = load_heat_points_from_html(HEATMAP_HTML_PATH)

@app.route("/accidents", methods=["GET"])
def accidents():
    """返回热力图点位 [[lat,lng], ...] 或 [[lat,lng,weight], ...]"""
    return jsonify(ACCIDENT_POINTS)

# 首页
@app.route("/")
def index():
    # 计算一个默认中心
    nodes = ox.graph_to_gdfs(G_wgs, edges=False)
    center = (float(nodes.geometry.y.mean()), float(nodes.geometry.x.mean()))
    return render_template("index.html", center_lat=center[0], center_lng=center[1], default_zoom=12)

# 计算路径（AJAX）
@app.route("/route", methods=["POST"])
def route():
    data = request.get_json()
    start = data["start"]  # [lat, lng]
    end   = data["end"]    # [lat, lng]
    beta  = float(data.get("beta", 0.1))

    s_node = ox.nearest_nodes(G_wgs, X=start[1], Y=start[0])
    e_node = ox.nearest_nodes(G_wgs, X=end[1],   Y=end[0])

    # --- 最短（按长度） ---
    p_short = nx.shortest_path(G_proj, source=s_node, target=e_node, weight="length")

    # --- 避险：为每条边写入 safe_cost 后，用字符串权重 ---
    for u, v, k, d in G_proj.edges(keys=True, data=True):
        length = float(d.get("length", 1.0))
        acc    = int(d.get("accident_count", 0))
        d["safe_cost"] = length + beta * acc * 10

    p_safe = nx.shortest_path(G_proj, source=s_node, target=e_node, weight="safe_cost")

    def path_len_acc(path):
        L, A = 0.0, 0
        for u, v in zip(path[:-1], path[1:]):
            ed = G_proj.get_edge_data(u, v)
            ed = list(ed.values())[0] if isinstance(ed, dict) and len(ed) else (ed or {})
            L += float(ed.get("length", 0.0))
            A += int(ed.get("accident_count", 0))
        return L, A

    def path_coords(path):
        # GeoJSON 需要 (lng, lat)
        return [(G_wgs.nodes[n]["x"], G_wgs.nodes[n]["y"]) for n in path]

    Ls, As = path_len_acc(p_short)
    Lr, Ar = path_len_acc(p_safe)

    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","properties":{"name":"short","length_m":Ls,"accidents_sum":As},
             "geometry":{"type":"LineString","coordinates": path_coords(p_short)}},
            {"type":"Feature","properties":{"name":"safe","length_m":Lr,"accidents_sum":Ar,"beta":beta},
             "geometry":{"type":"LineString","coordinates": path_coords(p_safe)}},
        ]
    }
    return jsonify(fc)


if __name__ == "__main__":
    app.run(debug=True)
