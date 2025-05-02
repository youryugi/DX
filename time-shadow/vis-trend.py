import pickle
from pprint import pprint

# 加载 pkl 文件
with open("edge_trend_map_20241205_0900_1000_1min_LL_135.5122_34.6246_UR_135.5502_34.6502.pkl", "rb") as f:
    trend_map = pickle.load(f)

# 显示前 5 项
for i, (key, value) in enumerate(trend_map.items()):
    print(f"{key}: {value}")
    if i >= 2000:
        break
