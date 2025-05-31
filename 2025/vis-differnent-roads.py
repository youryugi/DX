import osmnx as ox
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Yu Gothic'  # 支持日文显示

# 神户 六甲道区域（放大后）
north, south, east, west = 34.745, 34.70, 135.25, 135.225
bbox = (north, south, east, west)



# 提取路网（按 drivable 类型，若想包括更多类型可设为 'all_private'）
G = ox.graph_from_bbox(north, south, east, west, network_type='all')
edges = ox.graph_to_gdfs(G, nodes=False)

# 可视化
fig, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.6, zorder=1)


# 图例与标题
plt.legend(loc='lower right', fontsize=9)
plt.title("神戸・六甲道 地区用途別マップ＋路网", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()
