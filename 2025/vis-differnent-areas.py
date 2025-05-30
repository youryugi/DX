import osmnx as ox
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Yu Gothic'  # 支持日文显示
import matplotlib.patches as mpatches

# ...（你原来的绘图代码）...



# 神户 六甲道区域（放大后）
north, south, east, west = 34.745, 34.70, 135.252, 135.22
bbox = (north, south, east, west)

# 标签定义
tags_dict = {
    "station": {"railway": "station"},
    "school": {"amenity": "school"},
    "commercial": {"landuse": "commercial"},
    "residential": {"landuse": "residential"},
    "industrial": {"landuse": "industrial"},
    "office": {"building": "office"},
    "park": {"leisure": "park"},
    "tourism": {"tourism": True}
}

# 颜色与图例标签
color_map = {
    "station": "red",
    "school": "orange",
    "commercial": "blue",
    "residential": "green",
    "industrial": "gray",
    "office": "purple",
    "park": "lime",
    "tourism": "deeppink"
}
label_map = {
    "station": "駅",
    "school": "学校",
    "commercial": "商業地",
    "residential": "住宅地",
    "industrial": "工場",
    "office": "オフィス",
    "park": "公園",
    "tourism": "観光地"
}

# 提取 POI 类别数据
categories = {}
for key, tag in tags_dict.items():
    print('zhengzai..')
    gdf = ox.features.features_from_bbox(bbox=bbox, tags=tag)
    if not gdf.empty:
        categories[key] = gdf

# 提取路网（按 drivable 类型，若想包括更多类型可设为 'all_private'）
G = ox.graph_from_bbox(north, south, east, west, network_type='all')
edges = ox.graph_to_gdfs(G, nodes=False)

# 可视化
fig, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.6, zorder=1)

# 叠加不同类别的设施
for key, gdf in categories.items():
    gdf.plot(ax=ax, color=color_map[key], alpha=0.6, zorder=2)
# 构建 legend 手动条目
handles = []
for key in label_map:
    patch = mpatches.Patch(color=color_map[key], label=label_map[key])
    handles.append(patch)

# 替代 plt.legend()，使用手动 legend
ax.legend(handles=handles, loc='lower right', fontsize=9)
# 图例与标题
# plt.legend(loc='lower right', fontsize=9)
plt.title("神戸・六甲道 地区用途別マップ＋路网", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()
