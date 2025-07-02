import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import pandas as pd

# 读取建筑和道路数据
bldg_gml_files = [
    r"bldg\51357462_bldg_6697_op.gml",
    #r"bldg\51357451_bldg_6697_op.gml",
    # r"bldg\51357452_bldg_6697_op.gml",
    # r"bldg\51357453_bldg_6697_op.gml",
    # r"bldg\51357461_bldg_6697_op.gml",
    # r"bldg\51357462_bldg_6697_op.gml",
    # r"bldg\51357463_bldg_6697_op.gml",
    # r"bldg\51357471_bldg_6697_op.gml",
    # r"bldg\51357472_bldg_6697_op.gml",
    # r"bldg\51357473_bldg_6697_op.gml"
]
road_gml_files = [
    r"tran\51357462_tran_6697_op.gml",
    # r"tran\51357451_tran_6697_op.gml",
    # r"tran\51357452_tran_6697_op.gml",
    # r"tran\51357453_tran_6697_op.gml",
    # r"tran\51357461_tran_6697_op.gml",
    # r"tran\51357462_tran_6697_op.gml",
    # r"tran\51357463_tran_6697_op.gml",
    # r"tran\51357471_tran_6697_op.gml",
    # r"tran\51357472_tran_6697_op.gml",
    # r"tran\51357473_tran_6697_op.gml"
]

bldg_gdf_list = [gpd.read_file(file) for file in bldg_gml_files]
building_gdf = pd.concat(bldg_gdf_list, ignore_index=True)
road_gdf_list = [gpd.read_file(file) for file in road_gml_files]
road_gdf = pd.concat(road_gdf_list, ignore_index=True)

# 投影为EPSG:6669
if building_gdf.crs.to_epsg() != 6669:
    building_gdf = building_gdf.to_crs(epsg=6669)
if road_gdf.crs.to_epsg() != 6669:
    road_gdf = road_gdf.to_crs(epsg=6669)

building_gdf['usage'] = building_gdf['usage'].astype(str).str.strip()
usages = building_gdf['usage'].dropna().unique()
usages = [u for u in usages if u.lower() != 'nan']
cmap = plt.get_cmap('tab20', len(usages))
usage_color_map = {usage: mcolors.to_hex(cmap(i)) for i, usage in enumerate(usages)}
default_color = '#cccccc'
building_gdf['color'] = building_gdf['usage'].apply(lambda u: usage_color_map.get(u, default_color))

# 绘图
fig, ax = plt.subplots(figsize=(12, 8))
road_gdf.plot(ax=ax, color='black', linewidth=1, alpha=0.5, label='Road')
building_gdf.plot(ax=ax, color=building_gdf['color'])

# 图例
legend_handles = [Patch(facecolor=color, label=usage) for usage, color in usage_color_map.items()]
legend_handles.append(Patch(facecolor=default_color, label='Other'))
legend_handles.append(Patch(facecolor='black', label='Road'))
plt.legend(handles=legend_handles, loc='upper right', fontsize=12)

plt.title("Buildings by Usage and Roads", fontsize=16)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.show()