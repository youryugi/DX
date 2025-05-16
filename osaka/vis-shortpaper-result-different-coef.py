import matplotlib.pyplot as plt
import numpy as np
bigfontsize=14
# 数据准备
labels = ['α=0\n(Shortest route) ', 'α=0.1', 'α=0.2', 'α = 0.3 to 0.9 \n(step 0.1)', 'α=1']
sunny_lengths = [860, 570, 507, 504, 495]
shadow_lengths = [507, 825, 893, 897, 989]

# x 轴位置
bar_width = 0.3  # 明显缩小柱子宽度
x = np.arange(len(labels)) * 1.6  # 人为拉大间距，避免柱子太拥挤
# 创建图像
fig, ax = plt.subplots(figsize=(8, 5.5))

# 画阳光段（底部红色）
ax.bar(x, sunny_lengths, color='red', label='Sunny segment')

# 画阴影段（顶部绿色）
ax.bar(x, shadow_lengths, bottom=sunny_lengths, color='green', label='Shaded segment')
# 添加数值标签
# 添加内部数值标签
for i in range(len(x)):
    # 阳光段中部
    ax.text(x[i], sunny_lengths[i] / 2, f'{sunny_lengths[i]:.0f}', ha='center', va='center', fontsize=bigfontsize, color='white')
    # 阴影段中部（加上底部 sunny）
    ax.text(x[i], sunny_lengths[i] + shadow_lengths[i] / 2, f'{shadow_lengths[i]:.0f}', ha='center', va='center', fontsize=bigfontsize, color='white')

# 设置标题和标签
ax.set_ylabel('Distance (m)', fontsize=bigfontsize)
#ax.set_title('Comparison of Route Lengths in Sun and Shade', fontsize=bigfontsize)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=bigfontsize)
ax.legend(ncol=2,fontsize=bigfontsize)
plt.yticks(fontsize=bigfontsize)
# 布局优化
plt.tight_layout()
plt.show()
