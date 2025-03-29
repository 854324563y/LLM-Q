import matplotlib
import matplotlib.pyplot as plt
import numpy as np

zhfont1 = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf")

# 设置全局字体大小
plt.rcParams['font.size'] = 14
plt.style.use('seaborn-v0_8')  # 使用更现代的样式


# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

# 向量alpha由6个独热向量组成，每个独热向量\alpha^l表示一个线性模块的量化配置，alpha表示整个层的决策向量
alpha1 = [1,0,0]*6
alpha2 = [0,1,0]*6
alpha3 = [0,0,1]*6

# 创建图形和子图
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), height_ratios=[1, 1, 1])
axes = [ax1, ax2, ax3]
vectors = [alpha1, alpha2, alpha3]
names = ['α1', 'α2', 'α3']

# 设置子图间距
plt.subplots_adjust(hspace=0.4)

for ax, vec, name in zip(axes, vectors, names):
    # 将向量重塑为2D数组（1×18）
    vec_2d = np.array(vec).reshape(1, -1)
    
    # 创建热力图
    im = ax.imshow(vec_2d, aspect='auto', cmap='RdYlBu_r')
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加标签
    ax.set_ylabel(name, fontsize=16, rotation=0, ha='right', va='center')
    
    # 添加网格线以区分独热向量
    for i in range(1, 6):
        ax.axvline(x=i*3-0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    
    # 在每个独热向量中心添加数值标注
    for i in range(6):
        for j in range(3):
            val = vec[i*3 + j]
            if val == 1:
                ax.text(i*3 + j, 0, '1', ha='center', va='center', color='white', fontsize=12)

# 设置总标题
fig.suptitle('决策向量可视化\n(每3列构成一个独热向量)', fontproperties=zhfont1, fontsize=16, y=0.95)

# 保存图片，设置更高的DPI以获得更清晰的图像
plt.savefig('decision_vectors.png', dpi=300, bbox_inches='tight')
plt.close()



