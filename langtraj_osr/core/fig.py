import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 1. 全局字体与学术风格设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Arial', 'Helvetica', 'DejaVu Sans'], # 顶刊常用字体
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'mathtext.fontset': 'stix',
})

# 2. 提取数据
categories = ['≤10\n(98 users)', '11-20\n(38)', '21-30\n(16)', '>30\n(40)']
x = np.arange(len(categories))
auroc = [0.680, 0.738, 0.767, 0.792]

# 3. 创建画布 (高分辨率DPI=300标准)
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# 4. 设置莫兰迪浅色系 (低饱和度)
# 依次对应原图的红、浅蓝、深蓝、粉，但调整为更柔和的学术配色
colors = ['#E2998E', '#BCCBE0', '#90A3C1', '#DAB0B8']
edge_color = '#333333' # 柱子边框颜色较深，增加立体感

# 绘制柱状图，设置 zorder 确保柱子在网格线之上
bars = ax.bar(x, auroc, width=0.45, color=colors, edgecolor=edge_color, linewidth=1.0, zorder=3)

# 5. 绘制趋势线 (Trend)
ax.plot(x, auroc, color='#888888', linestyle='--', linewidth=1.5, zorder=2)
# 趋势线图例标注
ax.plot([-0.2, 0.0], [0.80, 0.80], color='#888888', linestyle='--', linewidth=1.5)
ax.text(0.05, 0.80, 'Trend (+0.112)', color='#444444', va='center', ha='left', fontsize=11)

# 6. 添加柱子上方的数据标签
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.003, f'{yval:.3f}', 
            ha='center', va='bottom', fontsize=10, color='#111111')

# 7. 添加红色虚线框与文字 (51% of users)
# 坐标点微调以完美包裹第一个柱子
box_x = x[0] - 0.28
box_y = 0.60
box_width = 0.56
box_height = 0.725 - 0.60

rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=1.2, 
                         edgecolor='#D9534F', facecolor='none', linestyle='--', zorder=4)
ax.add_patch(rect)
ax.text(x[0], 0.728, '51% of users', color='#D9534F', fontstyle='italic', 
        ha='center', va='bottom', fontsize=10)

# 8. 坐标轴、标题与网格优化
ax.set_ylim(0.60, 0.82) # 与原图保持一致的 Y 轴截断起点
ax.set_ylabel('AUROC', fontweight='bold')
ax.set_xlabel('User history length (# trips)', fontweight='bold')
ax.set_title('NYC: Performance vs. User History', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories)

# 仅保留水平网格线，透明度调低
ax.grid(axis='y', linestyle='-', alpha=0.2, color='gray', zorder=0)

# 去除顶部和右侧的边框 (Despine)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 9. 调整布局并输出
plt.tight_layout()
plt.show()