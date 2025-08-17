import pandas as pd
import folium
from folium.plugins import HeatMap

# ======== 配置 ========
CSV_PATH = r"kotsujiko2017-2024.csv"  # 改成你的CSV
LAT_COL  = "緯度（北緯）"    # 第24列（纬度）
LON_COL  = "経度（東経）"    # 第25列（经度）
OUTPUT_HTML = r"accident_heatmap_osm.html"

# 日本范围（清洗异常点）
LAT_MIN, LAT_MAX = 30.0, 46.5
LON_MIN, LON_MAX = 128.0, 146.5

# 热力参数
HEAT_RADIUS = 8
HEAT_BLUR   = 8
HEAT_MIN_OP = 0.4
HEAT_MAX_ZM = 16
# 1. HEAT_RADIUS
#
# 作用：每个点对周围产生的影响范围（单位是屏幕像素，而不是地图米数）。
#
# 数值越大 → 单个点的影响范围越广，热力会更“糊”、覆盖面积更大；多个点容易连成一片。
#
# 数值越小 → 热力图更“锐利”、点的范围更小、分布更稀疏。
#
# 建议：
#
# 城市级别（点很多） → 12~20 比较常用。
#
# 点非常密集 → 适当调小（8~12）。
#
# 点很稀疏 → 适当调大（20~30）。
#
# 2. HEAT_BLUR
#
# 作用：热力渲染的模糊程度，数值越大边缘越柔和。
#
# 数值越大 → 渐变平滑、热力过渡柔和，但可能显得不够精确。
#
# 数值越小 → 过渡更尖锐，热点更清晰。
#
# 建议：
#
# 一般与 HEAT_RADIUS 配合调节，常见比例是 BLUR ≈ RADIUS * 1.5 到 RADIUS * 2。
#
# 例：RADIUS=12 时，可以尝试 BLUR=15~24。
#
# 3. HEAT_MIN_OP (min_opacity)
#
# 作用：热力层的最低透明度。
#
# 数值越大 → 热力图整体更不透明，背景底图看得更少。
#
# 数值越小 → 热力图透明度高，背景底图更明显，但低密度区域会很淡甚至看不到。
#
# 建议：
#
# 如果想看清底图细节（道路等） → 0.3~0.5
#
# 如果只想强调热力，不太在意底图 → 0.5~0.8
#
# 4. HEAT_MAX_ZM (max_zoom)
#
# 作用：控制热力图的最大缩放等级影响范围（主要影响在高 zoom 时的绘制效果）。
#
# 一般设成 16~18 就够了，数值越大，放大地图时热力分布会按实际数据更细致地渲染。
#
# 如果设太小，放大到城市街道级时可能会看到热力“块状”分布。
# 可选自定义渐变（需要默认渐变则把 gradient=None）
GRADIENT = {
    0.0:  "#000000",
    0.2:  "#0000ff",
    0.4:  "#00ffff",
    0.6:  "#00ff00",
    0.8:  "#ffff00",
    1.0:  "#ff0000",
}

# --- 安全的底图模板（带 attribution） ---
TILES = {
    "OpenStreetMap": {
        "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": '© OpenStreetMap contributors'
    },
    # CartoDB Positron（清爽路网）
    "CartoDB Positron": {
        "tiles": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": '© OpenStreetMap contributors, © CARTO'
    },
    # CartoDB Voyager（路网更显眼）
    "CartoDB Voyager": {
        "tiles": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        "attr": '© OpenStreetMap contributors, © CARTO'
    },
    # CartoDB Dark Matter（深色底，热力对比强）
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

# 1) 读数据 + 清洗
df_all = read_csv_jp(CSV_PATH)
if LAT_COL not in df_all.columns or LON_COL not in df_all.columns:
    raise KeyError(f"未找到列：{LAT_COL} 或 {LON_COL}")

df = df_all[[LAT_COL, LON_COL]].dropna()
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
               control_scale=True)  # 显示比例尺（可选）

# 4) 再添加其它可切换底图（都带 attr，避免报错）
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

# 5) 热力层
heat_data = df[[LAT_COL, LON_COL]].values.tolist()
HeatMap(
    heat_data,
    radius=HEAT_RADIUS,
    blur=HEAT_BLUR,
    min_opacity=HEAT_MIN_OP,
    max_zoom=HEAT_MAX_ZM,
    gradient=GRADIENT  # 默认渐变用 gradient=None
).add_to(folium.FeatureGroup(name="事故热力图", show=True).add_to(m))

# 图层控制
folium.LayerControl(collapsed=False).add_to(m)

# 视野适配数据范围
m.fit_bounds(bounds, padding=(20, 20))

# 6) 保存
m.save(OUTPUT_HTML)
print(f"✅ 已生成：{OUTPUT_HTML}")
