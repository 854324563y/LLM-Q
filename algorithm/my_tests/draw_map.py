import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## 用于第四章敏感度矩阵的热力图

'''
zhfont1 = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf")

# with open('../log/hm/hm_0_Llama-2-7b-chat-hf.pkl', 'rb') as f:
#     q = pickle.load(f)
with open('log-adaptive/llama-7b-hf/hm_info_llama-7b-hf.pkl', 'rb') as f:
    q = pickle.load(f)[31]['hm_ori']

print(q.shape)  # (L,L)

# 创建热力图
plt.figure(figsize=(10, 8))
# 创建掩码来隐藏上三角部分
mask = np.triu(np.ones_like(q), k=1)
sns.heatmap(q, cmap='Reds', mask=mask)  # 使用红色渐变色系，并应用掩码
# plt.title('Quantization Sensitivity Map')
plt.title('量化误差交互矩阵')
plt.tight_layout()
plt.savefig('heatmap31.png')
plt.close()
'''

# 设置中文字体路径
zhfont1 = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf")

# 加载数据
with open('log-adaptive/llama-7b-hf/hm_info_llama-7b-hf.pkl', 'rb') as f:
    q = pickle.load(f)[0]['hm_ori']

print(q.shape)  # (L,L)

# 创建热力图
plt.figure(figsize=(10, 8))
# 创建掩码来隐藏上三角部分
mask = np.triu(np.ones_like(q), k=1)

# 绘制热力图
ax = sns.heatmap(q, cmap='Reds', mask=mask, annot=False, cbar_kws={'ticks': None})  # 使用红色渐变色系，并应用掩码

# 获取色标条对象并调整字体大小
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)  # 色标条字体大小设置为14

# 设置标题，使用中文字体并调整字体大小
plt.title('Llama-2-7b-第0层-量化误差交互矩阵', fontproperties=zhfont1, fontsize=20)  # 增大标题字体大小为16

# 调整坐标轴标签字体大小
plt.xticks(fontsize=14)  # 坐标轴刻度字体大小
plt.yticks(fontsize=14)

# 调整布局并保存图片
plt.tight_layout()
plt.savefig('heatmap0.png', dpi=300)  # 增加保存图片的分辨率
plt.close()