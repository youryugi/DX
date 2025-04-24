import geopandas as gpd
import pandas as pd

bldg_gml_files = [
    r"bldg\51357451_bldg_6697_op.gml",
    r"bldg\51357452_bldg_6697_op.gml",
    r"bldg\51357453_bldg_6697_op.gml",
    r"bldg\51357461_bldg_6697_op.gml",
    r"bldg\51357462_bldg_6697_op.gml",
    r"bldg\51357463_bldg_6697_op.gml",
    r"bldg\51357471_bldg_6697_op.gml",
    r"bldg\51357472_bldg_6697_op.gml",
    r"bldg\51357473_bldg_6697_op.gml"
]

bldg_gdf_list = [gpd.read_file(file) for file in bldg_gml_files]
bldg_merged_gdf = pd.concat(bldg_gdf_list, ignore_index=True)


# 2. 计算 WGS84 边界框并格式化为字符串
bounds = bldg_merged_gdf.to_crs(epsg=4326).total_bounds
minx, miny, maxx, maxy = bounds
minx_str = f"{minx:.4f}"
miny_str = f"{miny:.4f}"
maxx_str = f"{maxx:.4f}"
maxy_str = f"{maxy:.4f}"

# 3. 构造文件名并保存为 .pkl 文件
filename = f"bldg_merged_LL_{minx_str}_{miny_str}_UR_{maxx_str}_{maxy_str}.pkl"
save_path = rf"{filename}"
bldg_merged_gdf.to_pickle(save_path)

print(f"已保存至: {save_path}")
