import torch
import matplotlib.pyplot as plt
import numpy as np

layer_sensitivity_results1 = torch.load('../log/Llama-2-7b-chat-hf-w4a4/layer_sensitivity_results_lwtlet_ppl.pt')
layer_sensitivity_results2 = torch.load('../log/Llama-2-7b-chat-hf-w4a4/layer_sensitivity_results_nolwclet_ppl.pt')
full_precision_results = {'wikitext2': 7.076699256896973, 'c4': 8.918578147888184}

# 获取层数
layer_num = len(layer_sensitivity_results1)
layer_indices = np.arange(layer_num)

# 设置绘图样式
plt.figure(figsize=(12, 8))
colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
markers = ['o', 's']  # 圆形和方形
linestyles = ['-', '--']  # 实线和虚线

# 绘制results1的曲线
for idx, dataset in enumerate(layer_sensitivity_results1[0].keys()):
    ppl_scores = [layer_sensitivity_results1[i][dataset] for i in range(layer_num)]
    plt.plot(layer_indices, ppl_scores, 
             label=f'Quant-with-compensation-{dataset}',
             color=colors[0],
             marker=markers[idx],
             linestyle=linestyles[0],
             markersize=6,
             alpha=0.7)

# 绘制results2的曲线
for idx, dataset in enumerate(layer_sensitivity_results2[0].keys()):
    ppl_scores = [layer_sensitivity_results2[i][dataset] for i in range(layer_num)]
    plt.plot(layer_indices, ppl_scores, 
             label=f'Quant-{dataset}',
             color=colors[1],
             marker=markers[idx],
             linestyle=linestyles[1],
             markersize=6,
             alpha=0.7)

# 添加全精度结果的水平线
for dataset, ppl in full_precision_results.items():
    plt.axhline(y=ppl, color='gray', linestyle=':', label=f'Full Precision-{dataset}', alpha=0.5)

plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('PPL Score', fontsize=12)
plt.title('Layer-wise Quantization influence on PPL Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 保存图片
plt.savefig('layer_ppl_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

