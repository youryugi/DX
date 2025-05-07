import matplotlib.pyplot as plt
import numpy as np

# 数据准备
labels = ['Shortest\nRoute', 'Wanted\nα=1', 'Wanted\nα=0.9', 'Wanted\nα=0.2', 'Wanted\nα=0.1']
sunny_lengths = [859.5, 498.3, 503.9, 507.1, 568.5]
shadow_lengths = [507.2, 988.5, 897.3, 893.2, 824.5]

# x 轴位置
x = np.arange(len(labels))

# 创建图像
fig, ax = plt.subplots(figsize=(10, 6))

# 画阳光段（底部红色）
ax.bar(x, sunny_lengths, color='red', label='Sunny Segment')

# 画阴影段（顶部绿色）
ax.bar(x, shadow_lengths, bottom=sunny_lengths, color='green', label='Shaded Segment')

# 设置标题和标签
ax.set_ylabel('Distance (m)', fontsize=12)
ax.set_title('Comparison of Route Lengths in Sun and Shade', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)

# 布局优化
plt.tight_layout()
plt.show()
