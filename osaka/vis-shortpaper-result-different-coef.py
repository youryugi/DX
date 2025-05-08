import matplotlib.pyplot as plt
import numpy as np
bigfontsize=14
# 数据准备
labels = ['Shortest\nRoute', 'Wanted\nα=1', 'Wanted\nα=0.9', 'Wanted\nα=0.2', 'Wanted\nα=0.1']
sunny_lengths = [859.5, 498.3, 503.9, 507.1, 568.5]
shadow_lengths = [507.2, 988.5, 897.3, 893.2, 824.5]

labels = ['α=0\n(Shortest route) ', 'α=0.1', 'α=0.2', 'α = 0.3 to 0.9 \n(step 0.1)', 'α=1']
sunny_lengths = [859.5, 568.5, 507.1, 503.9, 498.3]
shadow_lengths = [507.2, 824.5, 893.2, 897.3, 988.5]


# x 轴位置

bar_width = 0.3  # 明显缩小柱子宽度
x = np.arange(len(labels)) * 1.5  # 人为拉大间距，避免柱子太拥挤
# 创建图像
fig, ax = plt.subplots(figsize=(8, 6))

# 画阳光段（底部红色）
ax.bar(x, sunny_lengths, color='red', label='Sunny segment')

# 画阴影段（顶部绿色）
ax.bar(x, shadow_lengths, bottom=sunny_lengths, color='green', label='Shaded segment')

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
