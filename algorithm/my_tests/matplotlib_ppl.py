import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

zhfont_small = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=18)
zhfont1 = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=22)
zhfont_large = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=24)


layer_sensitivity_results1 = torch.load('../log/Llama-2-7b-chat-hf-w4a4/layer_sensitivity_results_lwtlet_ppl.pt')
layer_sensitivity_results2 = torch.load('../log/Llama-2-7b-chat-hf-w4a4/layer_sensitivity_results_nolwclet_ppl.pt')
full_precision_results = {'wikitext2': 7.076699256896973, 'c4': 8.918578147888184}

def analyze_performance_improvement():
    # 获取层数
    layer_num = len(layer_sensitivity_results1)
    
    # 统计wikitext2数据集的性能提升情况
    improved_layers = 0
    total_ppl_with_comp = 0
    total_ppl_without_comp = 0
    
    for i in range(layer_num):
        ppl_with_comp = layer_sensitivity_results1[i]['wikitext2']
        ppl_without_comp = layer_sensitivity_results2[i]['wikitext2']
        
        if ppl_with_comp < ppl_without_comp:
            improved_layers += 1
            
        total_ppl_with_comp += ppl_with_comp
        total_ppl_without_comp += ppl_without_comp
    
    avg_ppl_with_comp = total_ppl_with_comp / layer_num
    avg_ppl_without_comp = total_ppl_without_comp / layer_num
    
    print(f"在wikitext2数据集上的分析结果：")
    print(f"总层数: {layer_num}")
    print(f"性能提升的层数: {improved_layers}")
    print(f"性能提升的层数占比: {improved_layers/layer_num*100:.2f}%")
    print(f"使用参数补偿的平均困惑度: {avg_ppl_with_comp:.4f}")
    print(f"不使用参数补偿的平均困惑度: {avg_ppl_without_comp:.4f}")
    print(f"平均困惑度提升: {avg_ppl_without_comp - avg_ppl_with_comp:.4f}")

def draw():
    # 获取层数
    layer_num = len(layer_sensitivity_results1)
    layer_indices = np.arange(layer_num)

    # 设置绘图样式
    plt.figure(figsize=(14, 10))
    # 为不同数据集和量化方式组合分配不同颜色
    colors = {
        'wikitext2-with-compensation': '#1f77b4',  # 蓝色
        'wikitext2-without-compensation': '#ff7f0e',  # 橙色
        'c4-with-compensation': '#2ca02c',  # 绿色
        'c4-without-compensation': '#d62728',  # 红色
    }
    markers = {'wikitext2': 'o', 'c4': 's'}  # 圆形和方形
    linestyles = {'with-compensation': '-', 'without-compensation': '--'}  # 实线和虚线

    # 绘制results1的曲线（with-compensation）
    for dataset in layer_sensitivity_results1[0].keys():
        ppl_scores = [layer_sensitivity_results1[i][dataset] for i in range(layer_num)]
        plt.plot(layer_indices, ppl_scores, 
                label=f'量化校准-参数补偿-{dataset}',
                color=colors[f'{dataset}-with-compensation'],
                marker=markers[dataset],
                linestyle=linestyles['with-compensation'],
                markersize=8,
                alpha=0.8)

    # 绘制results2的曲线（without-compensation）
    for dataset in layer_sensitivity_results2[0].keys():
        ppl_scores = [layer_sensitivity_results2[i][dataset] for i in range(layer_num)]
        plt.plot(layer_indices, ppl_scores, 
                label=f'量化校准-{dataset}',
                color=colors[f'{dataset}-without-compensation'],
                marker=markers[dataset],
                linestyle=linestyles['without-compensation'],
                markersize=8,
                alpha=0.8)

    # 将图例放在图内右上角
    plt.legend(loc='upper right', fontsize=18, prop=zhfont_small)


    # 添加全精度结果的水平线（使用灰色，避免与其他曲线颜色混淆）
    for dataset, ppl in full_precision_results.items():
        plt.axhline(y=ppl, color='gray', linestyle='--', 
                label=f'全精度-{dataset}', alpha=0.7, linewidth=2)
        # 在水平线下方添加注释，使用中文字体
        plt.text(31, ppl - 0.1, f'{dataset}全精度得分:{ppl:.4f}', 
                color='black', 
                verticalalignment='top', 
                horizontalalignment='right',
                fontproperties=zhfont_small)  # 使用之前定义的中文字体

    plt.xlabel('层索引', fontsize=22, fontproperties=zhfont1)
    plt.ylabel('困惑度分数', fontsize=22, fontproperties=zhfont1)
    plt.title('参数补偿对模型困惑度分数的影响', fontsize=24, fontproperties=zhfont_large)
    plt.grid(True, linestyle='--', alpha=0.3)


    # 设置x轴和y轴刻度的字体
    plt.xticks(fontsize=20, fontproperties=zhfont1)
    plt.yticks(fontsize=20, fontproperties=zhfont1)

    plt.tight_layout()

    # 保存图片
    plt.savefig('layer_ppl_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    analyze_performance_improvement()
    draw()