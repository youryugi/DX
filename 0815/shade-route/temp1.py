import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读四个txt文件"DSM_05OG701_1g.txt",
files = ["DSM_05OF893_1g.txt", "DSM_05OF894_1g.txt"]
dfs = [pd.read_csv(f, delim_whitespace=True, header=None, names=["x","y","z"]) for f in files]

# 合并
df = pd.concat(dfs, ignore_index=True)

# 构造网格（假设是规则格网）
x_unique = np.sort(df["x"].unique())
y_unique = np.sort(df["y"].unique())
X, Y = np.meshgrid(x_unique, y_unique)

# 根据 (x,y) 定位 z 值
Z = df.pivot_table(index="y", columns="x", values="z").values

# 画3D表面
fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="terrain", linewidth=0, antialiased=False)
plt.show()
