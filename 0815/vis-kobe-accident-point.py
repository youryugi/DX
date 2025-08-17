import pandas as pd
import folium
from folium.plugins import MarkerCluster

# ======== 配置 ========
CSV_PATH = r"kotsujiko2017-2024.csv"  # 改成你的CSV路径
LAT_COL  = "緯度（北緯）"    # 第24列（纬度）
LON_COL  = "経度（東経）"    # 第25列（经度）
OUTPUT_HTML = r"accident_points_osm.html"

# 仅展示你关心的字段（顺序即弹窗显示顺序；想增减就改这里）
FIELDS_TO_SHOW = [
    "警察署", "市区町", "発生年月日", "発生時間",
    "事故内容", "死者数", "負傷者数",
    "天候", "路面状態", "道路形状", "信号機", "道路線形",
    "事故類型",
    "年齢(当事者1)", "当事者種別(当事者1)",
    "年齢(当事者2)", "当事者種別(当事者2)",
    "中央分離施設",
]
FIELDS_TO_SHOW = [
    "発生年月日", "発生時間",
    "事故内容", "死者数", "負傷者数",
    "天候", "路面状態", "道路形状", "信号機",
    "事故類型",
]

# 日本范围（清洗异常点）
LAT_MIN, LAT_MAX = 30.0, 46.5
LON_MIN, LON_MAX = 128.0, 146.5

# --- 安全的底图模板（带 attribution） ---
TILES = {
    "OpenStreetMap": {
        "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": '© OpenStreetMap contributors'
    },
    "CartoDB Positron": {
        "tiles": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": '© OpenStreetMap contributors, © CARTO'
    },
    "CartoDB Voyager": {
        "tiles": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        "attr": '© OpenStreetMap contributors, © CARTO'
    },
    "CartoDB DarkMatter": {
        "tiles": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attr": '© OpenStreetMap contributors, © CARTO'
    },
}

def read_csv_jp(path):
    try:
        return pd.read_csv(path, encoding="shift_jis")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")

def color_for_row(row):
    try:
        deaths = int(row.get("死者数", 0) or 0)
    except:
        deaths = 0
    try:
        injuries = int(row.get("負傷者数", 0) or 0)
    except:
        injuries = 0

    if deaths > 0:
        return "red"
    elif injuries >= 2:
        return "orange"
    else:
        return "blue"

def make_popup_html(row, fields):
    # 只显示存在于CSV且这一行有值的字段
    items = []
    for f in fields:
        if f in row and pd.notna(row[f]) and str(row[f]).strip() != "":
            items.append(f"<tr><th style='text-align:left;white-space:nowrap;padding-right:8px;'>{f}</th>"
                         f"<td style='text-align:left;'>{row[f]}</td></tr>")
    table = "<table style='font-size:12px;border-collapse:collapse;'>" + "".join(items) + "</table>"
    return table

# 1) 读数据 + 清洗
df_all = read_csv_jp(CSV_PATH)
if LAT_COL not in df_all.columns or LON_COL not in df_all.columns:
    raise KeyError(f"未找到列：{LAT_COL} 或 {LON_COL}")

# 选择经纬度 + 弹窗可能用到的列（不存在的会自动跳过）
need_cols = list({LAT_COL, LON_COL, *FIELDS_TO_SHOW})
need_cols = [c for c in need_cols if c in df_all.columns]
df = df_all[need_cols].dropna(subset=[LAT_COL, LON_COL]).copy()

# 经纬度范围过滤
df = df[df[LAT_COL].between(LAT_MIN, LAT_MAX) & df[LON_COL].between(LON_MIN, LON_MAX)]
if df.empty:
    raise ValueError("过滤后无数据，请检查经纬度列与范围。")

# 2) 地图中心与范围
center = [df[LAT_COL].mean(), df[LON_COL].mean()]
bounds = [[df[LAT_COL].min(), df[LON_COL].min()],
          [df[LAT_COL].max(), df[LON_COL].max()]]

# 3) 初始化地图（先放一个 OSM 底图）
m = folium.Map(location=center, zoom_start=11,
               tiles=TILES["OpenStreetMap"]["tiles"],
               attr=TILES["OpenStreetMap"]["attr"],
               control_scale=True)

# 4) 添加其它可切换底图
for name, cfg in TILES.items():
    if name == "OpenStreetMap":
        continue
    folium.TileLayer(
        tiles=cfg["tiles"],
        name=name,
        attr=cfg["attr"],
        overlay=False,
        control=True
    ).add_to(m)

# 5) 添加事故点（聚合）
fg_points = folium.FeatureGroup(name="事故点", show=True)
cluster = MarkerCluster(name="事故点(聚合)", show=True)
for _, row in df.iterrows():
    lat = float(row[LAT_COL])
    lon = float(row[LON_COL])

    color = color_for_row(row)
    # 简短 Tooltip：地点 + 日期 + 时间 + 伤亡
    tip_parts = []
    for key in ["市区町", "発生年月日", "発生時間", "事故内容", "死者数", "負傷者数"]:
        if key in row and pd.notna(row[key]):
            tip_parts.append(f"{key}:{row[key]}")
    tooltip_txt = " | ".join(tip_parts) if tip_parts else "事故情報"

    popup_html = make_popup_html(row, FIELDS_TO_SHOW)
    marker = folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color=color,
        fill=True,
        fill_opacity=0.8,
        weight=1,
        tooltip=tooltip_txt,
        popup=folium.Popup(popup_html, max_width=380, min_width=260),
    )
    cluster.add_child(marker)

fg_points.add_child(cluster).add_to(m)

# 6) 图层控制 + 视野
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(bounds, padding=(20, 20))

# 7) 保存
m.save(OUTPUT_HTML)
print(f"✅ 已生成：{OUTPUT_HTML}")
