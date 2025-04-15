import geopandas as gpd
import pandas as pd

bldg_gml_files = [
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357451_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357452_bldg_6697_op.gml",
    r"C:\Users\79152\Desktop\OthersProgramme\DX\time-shadow\bldg\51357453_bldg_6697_op.gml"
]

bldg_gdf_list = [gpd.read_file(file) for file in bldg_gml_files]
bldg_merged_gdf = pd.concat(bldg_gdf_list, ignore_index=True)

# 保存为GeoJSON
bldg_merged_gdf.to_file(r"C:\Users\79152\Desktop\OthersProgramme\DX\paper\merged_bldg.geojson", driver='GeoJSON')
