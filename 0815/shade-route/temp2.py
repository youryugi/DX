import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Transformer
import osmnx as ox
from shapely.geometry import LineString
from scipy.interpolate import griddata
from tqdm import tqdm   # pip install tqdm
# === 1) 读 & 合并 DSM 点 ===
files = ["DSM_05OF892_1g.txt"]
dfs = [pd.read_csv(f, delim_whitespace=True, header=None, names=["x","y","z"]) for f in files]
df = pd.concat(dfs, ignore_index=True)

# === 2) 构造规则网格（若你的数据本来就是规则格网）===
x_unique = np.sort(df["x"].unique())
y_unique = np.sort(df["y"].unique())
X, Y = np.meshgrid(x_unique, y_unique)
# pivot 成 Z 栅格（行=Y从小到大）
Z = df.pivot_table(index="y", columns="x", values="z").reindex(index=y_unique, columns=x_unique).values

# DSM 范围（投影坐标）
xmin, xmax = x_unique[0], x_unique[-1]
ymin, ymax = y_unique[0], y_unique[-1]

# === 3) 用 pyproj 把 DSM 范围转成 WGS84，便于 OSMnx 抓路网 ===
# 替换为你的 DSM EPSG
EPSG_DSM = 2447  # <-- TODO: 换成你的DSM投影EPSG
transform_to_wgs84 = Transformer.from_crs(EPSG_DSM, 4326, always_xy=True)
# 取四角并求 bbox
lon_min, lat_min = transform_to_wgs84.transform(xmin, ymin)
lon_max, lat_max = transform_to_wgs84.transform(xmax, ymax)

# === 4) 抓取 OSM 路网（WGS84）===
G = ox.graph_from_bbox(north=lat_max, south=lat_min, east=lon_max, west=lon_min, network_type="drive")  # 或 'all_private'/'walk' 等
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# === 5) 路网坐标投回 DSM 投影 ===
transform_to_dsm = Transformer.from_crs(4326, EPSG_DSM, always_xy=True)

def reproject_linestring(ls: LineString):
    xs, ys = ls.xy
    x2, y2 = transform_to_dsm.transform(np.array(xs), np.array(ys))
    return LineString(np.column_stack([x2, y2]))

gdf_edges_proj = gdf_edges.copy()
gdf_edges_proj["geometry"] = gdf_edges_proj["geometry"].apply(reproject_linestring)

# === 6A) 2D 叠加：imshow DSM + plot 路网 ===
plt.figure(figsize=(10,10))
# 注意 imshow 需要指定 extent，且 origin='lower' 让坐标正向一致
plt.imshow(Z, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='terrain')
for geom in gdf_edges_proj.geometry:
    xs, ys = geom.xy
    plt.plot(xs, ys, linewidth=0.8)
plt.title("DSM + Road network (2D)")
plt.xlabel("X (projected)")
plt.ylabel("Y (projected)")
plt.show()
print("kaishi hua 3d")
# 画3D表面
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="terrain", linewidth=0, antialiased=False, alpha=0.95)

points = np.column_stack([X.ravel(), Y.ravel()])
values = Z.ravel()

def line3d_coords(ls: LineString):
    xs, ys = np.array(ls.xy[0]), np.array(ls.xy[1])
    zs = griddata(points, values, (xs, ys), method='nearest')
    return xs, ys, zs

# tqdm 显示进度
for geom in tqdm(gdf_edges_proj.geometry, desc="Plotting roads"):
    xs, ys, zs = line3d_coords(geom)
    ax.plot(xs, ys, zs, linewidth=0.8, color="black")

plt.show()
