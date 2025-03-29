import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

## 用于第四章自适应量化的得分图

wikitext2_fp16 = {
    'llama-7b-hf': 5.6770,
    'llama-13b-hf': 5.0907,
    'Llama-2-7b-hf': 5.4722,
    'Llama-2-13b-hf': 4.8836,
}

c4_fp16 = {
    'llama-7b-hf': 7.0790,
    'llama-13b-hf': 6.6113,
    'Llama-2-7b-hf': 6.9728,
    'Llama-2-13b-hf': 6.4682,
}

average_acc_fp16 = {
    'llama-7b-hf': 63.38,
    'llama-13b-hf': 65.83,
    'Llama-2-7b-hf': 63.80,
    'Llama-2-13b-hf': 65.99,
}

zhfont1 = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf")
zhfont_large = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=16)  # 大号中文字体
zhfont_extra = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=20)  # 特大号中文字体

def plot_quantization_results(csv_file, net_name, score_metric):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 过滤出指定网络的数据
    df = df[df['net'] == net_name]
    
    # 按 factor 排序
    df = df.sort_values(by='factor')
    
    # 设置 Seaborn 风格和字体大小
    plt.rcParams.update({'font.size': 16})
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制得分曲线
    color = 'tab:blue'
    ax1.set_xlabel(f'量化比因子($\\lambda$)', fontproperties=zhfont_extra, fontsize=20)
    ax1.set_ylabel(score_metric, color=color, fontproperties=zhfont_extra, fontsize=20)
    ax1.plot(df['factor'], df[score_metric], marker='o', color=color, label=score_metric, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    
    # 添加 FP16 基准线
    if score_metric == 'wikitext2':
        fp16_score = wikitext2_fp16[net_name]
    elif score_metric == 'c4':
        fp16_score = c4_fp16[net_name]
    elif score_metric == 'average_acc':
        fp16_score = average_acc_fp16[net_name]
    
    ax1.axhline(y=fp16_score, color=color, linestyle='--', alpha=0.5, label=f'FP16基准 ({fp16_score:.4f})', linewidth=2)
    # 在y轴右侧添加fp16得分标注
    ax1.text(df['factor'].max(), fp16_score, f'FP16基准: {fp16_score:.4f}', 
             color=color, va='bottom', ha='right', fontproperties=zhfont_large, fontsize=16)
    
    # 创建共享 x 轴的第二个 y 轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('位运算量', color=color, fontproperties=zhfont_extra, fontsize=16)
    ax2.plot(df['factor'], df['bitops'], marker='s', linestyle='--', color=color, 
             label='位运算量', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)
    
    # 标题
    plt.title(f'{net_name} - {score_metric} vs. 位运算量', fontproperties=zhfont_extra, fontsize=16, pad=20)
    
    # 显示图例
    fig.tight_layout()
    ax1.legend(loc='upper left', prop=zhfont_large, fontsize=20)
    ax2.legend(loc='upper right', prop=zhfont_large, fontsize=20)
    
    # 设置更大的边距
    plt.subplots_adjust(right=0.85)
    
    # 显示图表
    plt.show()
    plt.savefig(f'my_tests/{net_name}_{score_metric}.png', dpi=300, bbox_inches='tight')

def plot_all_metrics(csv_file, net_name):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 过滤出指定网络的数据
    df = df[df['net'] == net_name]
    
    # 按 factor 排序
    df = df.sort_values(by='factor')
    
    # 设置 Seaborn 风格和字体大小
    plt.rcParams.update({'font.size': 16})  # 增大默认字体大小
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # 创建一行两列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # 左图：平均准确率
    color = 'tab:blue'
    ax1.set_xlabel(f'量化比因子($\\lambda$)', fontproperties=zhfont_extra, fontsize=18)
    ax1.set_ylabel('平均准确率', color=color, fontproperties=zhfont_extra, fontsize=18)
    line1_1 = ax1.plot(df['factor'], df['average_acc'], marker='o', color=color, 
             label='平均准确率', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    
    # 添加 FP16 基准线
    fp16_acc = average_acc_fp16[net_name]
    line1_2 = ax1.axhline(y=fp16_acc, color=color, linestyle='--', alpha=0.5, 
                label=f'FP16基准 ({fp16_acc:.4f})', linewidth=2)
    ax1.text(df['factor'].max(), fp16_acc, f'FP16基准: {fp16_acc:.4f}', 
             color=color, va='bottom', ha='right', fontproperties=zhfont_large, fontsize=16)
    
    # 左图添加位运算量
    ax1_twin = ax1.twinx()
    color_bitops = 'tab:red'
    ax1_twin.set_ylabel('位运算量', color=color_bitops, fontproperties=zhfont_extra, fontsize=16)
    line1_3 = ax1_twin.plot(df['factor'], df['bitops'], marker='s', linestyle='--', 
                 color=color_bitops, label='位运算量', linewidth=2, markersize=8)
    ax1_twin.tick_params(axis='y', labelcolor=color_bitops, labelsize=16)
    
    # 右图：困惑度指标
    color1 = 'tab:blue'
    color2 = 'tab:green'
    ax2.set_xlabel(f'量化比因子($\\lambda$)', fontproperties=zhfont_extra, fontsize=18)
    ax2.set_ylabel('困惑度', fontproperties=zhfont_extra, fontsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    
    line1 = ax2.plot(df['factor'], df['wikitext2'], marker='o', color=color1, 
                     label='WikiText-2困惑度', linewidth=2, markersize=8)
    line2 = ax2.plot(df['factor'], df['c4'], marker='^', color=color2, 
                     label='C4困惑度', linewidth=2, markersize=8)
    
    # 添加困惑度 FP16 基准线
    fp16_wiki = wikitext2_fp16[net_name]
    fp16_c4 = c4_fp16[net_name]
    
    ax2.axhline(y=fp16_wiki, color=color1, linestyle='--', alpha=0.5, 
                label=f'WikiText-2 FP16基准 ({fp16_wiki:.4f})', linewidth=2)
    ax2.axhline(y=fp16_c4, color=color2, linestyle='--', alpha=0.5, 
                label=f'C4 FP16基准 ({fp16_c4:.4f})', linewidth=2)
    
    ax2.text(df['factor'].max(), fp16_wiki, f'WikiText-2 FP16基准: {fp16_wiki:.4f}', 
             color=color1, va='top', ha='right', fontproperties=zhfont_large, fontsize=16)
    ax2.text(df['factor'].max(), fp16_c4, f'C4 FP16基准: {fp16_c4:.4f}', 
             color=color2, va='top', ha='right', fontproperties=zhfont_large, fontsize=16)
    
    # 右图添加位运算量
    ax2_twin = ax2.twinx()
    color_bitops = 'tab:red'
    ax2_twin.set_ylabel('位运算量', color=color_bitops, fontproperties=zhfont_extra, fontsize=16)
    line3 = ax2_twin.plot(df['factor'], df['bitops'], marker='s', linestyle='--', 
                         color=color_bitops, label='位运算量', linewidth=2, markersize=8)
    ax2_twin.tick_params(axis='y', labelcolor=color_bitops, labelsize=16)
    
    # 添加图例
    lines1 = line1_1 + [line1_2] + line1_3
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='upper left', prop=zhfont_large, fontsize=20)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', prop=zhfont_large, fontsize=16)
    
    # 设置总标题
    fig.suptitle(f'{net_name} - 自适应量化性能', fontproperties=zhfont_extra, fontsize=20, y=0.95)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'my_tests/{net_name}_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_perplexity_metrics(csv_file, net_name):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 过滤出指定网络的数据
    df = df[df['net'] == net_name]
    
    # 按 factor 排序
    df = df.sort_values(by='factor')
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 准备标签文本
    wiki_label = 'WikiText-2困惑度'
    c4_label = 'C4困惑度'
    bitops_label = '位运算量'
    wiki_baseline_label = f'WikiText-2 FP16基准 ({wikitext2_fp16[net_name]:.4f})'
    c4_baseline_label = f'C4 FP16基准 ({c4_fp16[net_name]:.4f})'
    
    # 绘制 wikitext2 得分曲线
    color1 = 'tab:blue'
    ax1.set_xlabel(f'量化比因子($\\lambda$)', fontproperties=zhfont_extra, fontsize=24)
    ax1.set_ylabel('困惑度', fontproperties=zhfont_extra, fontsize=24)
    line1 = ax1.plot(df['factor'], df['wikitext2'], marker='o', color=color1, 
                     label=wiki_label, linewidth=2, markersize=8)
    
    # 设置刻度字体大小
    ax1.tick_params(axis='both', labelsize=20)
    
    # 绘制 c4 得分曲线
    color2 = 'tab:green'
    line2 = ax1.plot(df['factor'], df['c4'], marker='^', color=color2, 
                     label=c4_label, linewidth=2, markersize=8)
    
    # 添加 FP16 基准线
    fp16_wiki = wikitext2_fp16[net_name]
    fp16_c4 = c4_fp16[net_name]
    
    ax1.axhline(y=fp16_wiki, color=color1, linestyle='--', alpha=0.5, 
                label=wiki_baseline_label, linewidth=2)
    ax1.axhline(y=fp16_c4, color=color2, linestyle='--', alpha=0.5, 
                label=c4_baseline_label, linewidth=2)
    
    # 在y轴右侧添加fp16得分标注
    ax1.text(df['factor'].max(), fp16_wiki, f'WikiText-2 FP16基准: {fp16_wiki:.4f}', 
             color=color1, va='top', ha='right', fontproperties=zhfont_large, fontsize=20)
    ax1.text(df['factor'].max(), fp16_c4, f'C4 FP16基准: {fp16_c4:.4f}', 
             color=color2, va='top', ha='right', fontproperties=zhfont_large, fontsize=20)
    
    # 创建共享 x 轴的第二个 y 轴
    ax2 = ax1.twinx()
    color3 = 'tab:red'
    ax2.set_ylabel('位运算量', color=color3, fontproperties=zhfont_extra, fontsize=24)
    line3 = ax2.plot(df['factor'], df['bitops'], marker='s', linestyle='--', 
                     color=color3, label=bitops_label, linewidth=2, markersize=8)
    
    # 设置第二个y轴的刻度字体
    ax2.tick_params(axis='y', labelcolor=color3, labelsize=20)
    
    # 合并所有线条的图例
    lines = line1 + line2 + line3
    labels = [wiki_label, c4_label, bitops_label]
    legend = ax1.legend(lines, labels, loc='upper left', prop=zhfont_large, fontsize=20)
    
    # 标题
    plt.title(f'{net_name} - 困惑度指标与位运算量对比', fontproperties=zhfont_extra, fontsize=24, pad=20)
    
    # 设置更大的边距
    plt.subplots_adjust(right=0.85)
    
    # 保存图表
    plt.savefig(f'my_tests/{net_name}_perplexity.png', dpi=300, bbox_inches='tight')
    plt.show()

# 示例调用
# plot_quantization_results('my_tests/adaptive-table.csv', 'llama-7b-hf', 'wikitext2')
# plot_quantization_results('my_tests/adaptive-table.csv', 'llama-7b-hf', 'c4')
# plot_quantization_results('my_tests/adaptive-table.csv', 'llama-7b-hf', 'average_acc')

# llama-13b-hf  Llama-2-13b-hf
plot_all_metrics('my_tests/adaptive-table.csv', 'llama-13b-hf')

# 示例调用
#plot_perplexity_metrics('my_tests/adaptive-table.csv', 'Llama-2-7b-hf')
