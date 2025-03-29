import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

# 字体设置
zhfont1 = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf")
zhfont_large = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=16)  # 大号中文字体
zhfont_extra = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=20)  # 特大号中文字体

# FP16基准值参考
wikitext2_fp16 = {
    'llama-7b': 5.6770,
    'llama-13b': 5.0907,
    'Llama-2-7b': 5.4722,
    'Llama-2-13b': 4.8836,
}

c4_fp16 = {
    'llama-7b': 7.0790,
    'llama-13b': 6.6113,
    'Llama-2-7b': 6.9728,
    'Llama-2-13b': 6.4682,
}

average_acc_fp16 = {
    'llama-7b': 63.38,
    'llama-13b': 65.83,
    'Llama-2-7b': 63.80,
    'Llama-2-13b': 65.99,
}

def compare_block_layer_metrics(xlsx_file, net_name, metric):
    """
    比较逐块量化和逐层量化在指定指标上的性能
    
    参数:
    xlsx_file: Excel文件路径
    net_name: 模型名称
    metric: 要比较的指标 ('wikitext2', 'c4', 'average_acc')
    """
    # 读取Excel文件中的两个表单
    df_block = pd.read_excel(xlsx_file, sheet_name='block')
    df_layer = pd.read_excel(xlsx_file, sheet_name='layer')
    
    # 过滤出指定网络的数据
    df_block = df_block[df_block['net'] == net_name].sort_values(by='factor')
    df_layer = df_layer[df_layer['net'] == net_name].sort_values(by='factor')
    
    # 设置Seaborn风格
    plt.rcParams.update({'font.size': 16})
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 准备标签文本
    block_label = '逐块量化'
    layer_label = '逐层量化'
    bitops_block_label = '逐块量化位运算量'
    bitops_layer_label = '逐层量化位运算量'
    
    # 获取指标的中文名称
    if metric == 'wikitext2':
        metric_cn = 'WikiText-2困惑度'
        fp16_baseline = wikitext2_fp16[net_name]
    elif metric == 'c4':
        metric_cn = 'C4困惑度'
        fp16_baseline = c4_fp16[net_name]
    elif metric == 'average_acc':
        metric_cn = '平均准确率'
        fp16_baseline = average_acc_fp16[net_name]
    
    # 绘制逐块量化的曲线
    color1 = 'tab:blue'
    ax1.set_xlabel(f'量化比因子($\\lambda$)', fontproperties=zhfont_extra, fontsize=24)
    ax1.set_ylabel(metric_cn, fontproperties=zhfont_extra, fontsize=24)
    line1 = ax1.plot(df_block['factor'], df_block[metric], marker='o', color=color1, 
                     label=f'{block_label}-{metric_cn}', linewidth=2, markersize=8)
    
    # 绘制逐层量化的曲线
    color2 = 'tab:green'
    line2 = ax1.plot(df_layer['factor'], df_layer[metric], marker='^', color=color2,
                     label=f'{layer_label}-{metric_cn}', linewidth=2, markersize=8)
    
    # 设置刻度字体大小
    ax1.tick_params(axis='both', labelsize=20)
    
    # 添加FP16基准线
    ax1.axhline(y=fp16_baseline, color='black', linestyle='--', alpha=0.7,
                label=f'FP16基准 ({fp16_baseline:.4f})', linewidth=2)
    
    # 在y轴右侧添加fp16得分标注 - 避免使用可能包含NaN或Inf的值
    max_factor = max(df_block['factor'].max(), df_layer['factor'].max())
    if np.isfinite(max_factor) and np.isfinite(fp16_baseline):
        ax1.text(max_factor, fp16_baseline, 
                f'FP16基准: {fp16_baseline:.4f}', color='black', va='bottom', ha='right', 
                fontproperties=zhfont_large, fontsize=20)
    
    # 创建共享x轴的第二个y轴
    ax2 = ax1.twinx()
    color3 = 'tab:red'
    color4 = 'tab:purple'
    ax2.set_ylabel('位运算量', color='black', fontproperties=zhfont_extra, fontsize=24)
    
    # 绘制逐块量化的位运算量
    line3 = ax2.plot(df_block['factor'], df_block['bitops'], marker='s', linestyle='--', 
                     color=color3, label=bitops_block_label, linewidth=2, markersize=8)
    
    # 绘制逐层量化的位运算量
    line4 = ax2.plot(df_layer['factor'], df_layer['bitops'], marker='d', linestyle='--', 
                     color=color4, label=bitops_layer_label, linewidth=2, markersize=8)
    
    # 设置第二个y轴的刻度字体
    ax2.tick_params(axis='y', labelsize=20)
    
    # 合并所有线条的图例
    lines = line1 + line2 + line3 + line4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', prop=zhfont_large, fontsize=20)
    
    # 标题
    plt.title(f'{net_name} - 逐块与逐层量化{metric_cn}对比', fontproperties=zhfont_extra, fontsize=24, pad=20)
    
    # 设置更大的边距
    plt.subplots_adjust(right=0.85)
    
    # 保存图表
    plt.savefig(f'my_tests/{net_name}_{metric}_block_vs_layer.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_comprehensive_metrics(xlsx_file, net_name):
    """
    综合比较逐块量化和逐层量化在多个指标上的性能
    
    参数:
    xlsx_file: Excel文件路径
    net_name: 模型名称
    """
    # 读取Excel文件中的两个表单
    df_block = pd.read_excel(xlsx_file, sheet_name='block')
    df_layer = pd.read_excel(xlsx_file, sheet_name='layer')
    
    # 过滤出指定网络的数据
    df_block = df_block[df_block['net'] == net_name].sort_values(by='factor')
    df_layer = df_layer[df_layer['net'] == net_name].sort_values(by='factor')
    
    # 设置Seaborn风格
    plt.rcParams.update({'font.size': 16})
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # 创建一行两列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左图：平均准确率比较
    color1 = 'tab:blue'
    color2 = 'tab:green'
    ax1.set_xlabel(f'量化比因子($\\lambda$)', fontproperties=zhfont_extra, fontsize=18)
    ax1.set_ylabel('平均准确率', fontproperties=zhfont_extra, fontsize=18)
    
    # 绘制逐块和逐层的平均准确率
    line1_1 = ax1.plot(df_block['factor'], df_block['average_acc'], marker='o', color=color1, 
                     label='逐块量化-平均准确率', linewidth=2, markersize=8)
    line1_2 = ax1.plot(df_layer['factor'], df_layer['average_acc'], marker='^', color=color2, 
                     label='逐层量化-平均准确率', linewidth=2, markersize=8)
    
    ax1.tick_params(axis='both', labelsize=16)
    
    # 添加FP16基准线
    fp16_acc = average_acc_fp16[net_name]
    ax1.axhline(y=fp16_acc, color='black', linestyle='--', alpha=0.7, 
                label=f'FP16基准 ({fp16_acc:.2f})', linewidth=2)
    
    # 在y轴右侧添加fp16得分标注 - 检查并确保值是有限的
    max_factor = max(df_block['factor'].max(), df_layer['factor'].max())
    if np.isfinite(max_factor) and np.isfinite(fp16_acc):
        ax1.text(max_factor, fp16_acc + 0.1, 
                f'FP16基准: {fp16_acc:.2f}', color='black', va='bottom', ha='right', 
                fontproperties=zhfont_large, fontsize=16)
    
    # 左图添加位运算量
    ax1_twin = ax1.twinx()
    color3 = 'tab:red'
    color4 = 'tab:purple'
    ax1_twin.set_ylabel('位运算量', color='black', fontproperties=zhfont_extra, fontsize=16)
    
    # 绘制逐块和逐层的位运算量
    line1_3 = ax1_twin.plot(df_block['factor'], df_block['bitops'], marker='s', linestyle='--', 
                          color=color3, label='逐块量化-位运算量', linewidth=2, markersize=8)
    line1_4 = ax1_twin.plot(df_layer['factor'], df_layer['bitops'], marker='d', linestyle='--', 
                          color=color4, label='逐层量化-位运算量', linewidth=2, markersize=8)
    
    ax1_twin.tick_params(axis='y', labelsize=16)
    
    # 右图：困惑度指标比较
    ax2.set_xlabel(f'量化比因子($\\lambda$)', fontproperties=zhfont_extra, fontsize=18)
    ax2.set_ylabel('困惑度', fontproperties=zhfont_extra, fontsize=18)
    ax2.tick_params(axis='both', labelsize=16)
    
    # 绘制逐块和逐层的WikiText-2困惑度
    line2_1 = ax2.plot(df_block['factor'], df_block['wikitext2'], marker='o', color=color1, 
                     label='逐块量化-WikiText-2困惑度', linewidth=2, markersize=8)
    line2_2 = ax2.plot(df_layer['factor'], df_layer['wikitext2'], marker='^', color=color2, 
                     label='逐层量化-WikiText-2困惑度', linewidth=2, markersize=8)
    
    # 添加困惑度FP16基准线
    fp16_wiki = wikitext2_fp16[net_name]
    
    ax2.axhline(y=fp16_wiki, color='black', linestyle='--', alpha=0.5, linewidth=2) # color=color1
    
    # 添加标签文本，确保坐标是有限的
    if np.isfinite(max_factor) and np.isfinite(fp16_wiki):
        ax2.text(max_factor, fp16_wiki - 0.05, 
                f'WikiText-2 FP16基准: {fp16_wiki:.4f}', color='black', va='top', ha='right',  # color=color1
                fontproperties=zhfont_large, fontsize=16)
    
    # 右图添加位运算量
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('位运算量', color='black', fontproperties=zhfont_extra, fontsize=16)
    
    # 绘制逐块和逐层的位运算量
    line2_5 = ax2_twin.plot(df_block['factor'], df_block['bitops'], marker='s', linestyle='--', 
                          color=color3, label='逐块量化-位运算量', linewidth=2, markersize=8)
    line2_6 = ax2_twin.plot(df_layer['factor'], df_layer['bitops'], marker='d', linestyle='--', 
                          color=color4, label='逐层量化-位运算量', linewidth=2, markersize=8)
    
    ax2_twin.tick_params(axis='y', labelsize=16)
    
    # 添加图例
    # 左图图例
    lines1 = line1_1 + line1_2 + line1_3 + line1_4
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='upper left', prop=zhfont_large, fontsize=14)
    
    # 右图图例
    lines2 = line2_1 + line2_2 + line2_5 + line2_6
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='upper left', prop=zhfont_large, fontsize=14)
    
    # 设置总标题
    fig.suptitle(f'{net_name} - 逐块量化与逐层量化性能对比', fontproperties=zhfont_extra, fontsize=20, y=0.98)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'my_tests/{net_name}_comprehensive_block_vs_layer.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_relative_improvement(xlsx_file, net_name):
    """
    绘制逐块量化相对于逐层量化的性能提升百分比
    
    参数:
    xlsx_file: Excel文件路径
    net_name: 模型名称
    """
    # 读取Excel文件中的两个表单
    df_block = pd.read_excel(xlsx_file, sheet_name='block')
    df_layer = pd.read_excel(xlsx_file, sheet_name='layer')
    
    # 过滤出指定网络的数据
    df_block = df_block[df_block['net'] == net_name].sort_values(by='factor')
    df_layer = df_layer[df_layer['net'] == net_name].sort_values(by='factor')
    
    # 确保两个数据集有相同的factor值
    common_factors = sorted(set(df_block['factor']).intersection(set(df_layer['factor'])))
    df_block = df_block[df_block['factor'].isin(common_factors)]
    df_layer = df_layer[df_layer['factor'].isin(common_factors)]
    
    # 重新排序确保对齐
    df_block = df_block.sort_values(by='factor')
    df_layer = df_layer.sort_values(by='factor')
    
    # 计算性能改进百分比
    improvements = pd.DataFrame()
    improvements['factor'] = common_factors
    
    # 使用安全的除法函数避免除以零
    def safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    # 困惑度指标（越低越好）- 负值表示提升
    # 使用安全除法避免除以零或极小值
    wikitext2_block = df_block['wikitext2'].values
    wikitext2_layer = df_layer['wikitext2'].values
    c4_block = df_block['c4'].values
    c4_layer = df_layer['c4'].values
    acc_block = df_block['average_acc'].values
    acc_layer = df_layer['average_acc'].values
    bitops_block = df_block['bitops'].values
    bitops_layer = df_layer['bitops'].values
    
    # 使用安全除法计算改进百分比，并处理可能的无限值
    wiki_imp = safe_div(wikitext2_block - wikitext2_layer, wikitext2_layer) * 100
    c4_imp = safe_div(c4_block - c4_layer, c4_layer) * 100
    acc_imp = safe_div(acc_block - acc_layer, acc_layer) * 100
    bitops_imp = safe_div(bitops_block - bitops_layer, bitops_layer) * 100
    
    # 替换可能的无限值和NaN值
    wiki_imp = np.nan_to_num(wiki_imp, nan=0.0, posinf=0.0, neginf=0.0)
    c4_imp = np.nan_to_num(c4_imp, nan=0.0, posinf=0.0, neginf=0.0)
    acc_imp = np.nan_to_num(acc_imp, nan=0.0, posinf=0.0, neginf=0.0)
    bitops_imp = np.nan_to_num(bitops_imp, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(wiki_imp, sum(wiki_imp)/len(wiki_imp))
    print(acc_imp, sum(acc_imp)/len(acc_imp))

    improvements['wikitext2_improvement'] = wiki_imp
    improvements['c4_improvement'] = c4_imp
    improvements['accuracy_improvement'] = acc_imp
    improvements['bitops_improvement'] = bitops_imp
    
    # 设置Seaborn风格
    plt.rcParams.update({'font.size': 16})
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制性能改进曲线
    ax.plot(improvements['factor'], improvements['wikitext2_improvement'], marker='o', label='WikiText-2困惑度改进', linewidth=2, markersize=8)
    ax.plot(improvements['factor'], improvements['accuracy_improvement'], marker='s', label='平均准确率改进', linewidth=2, markersize=8)
    
    # 添加0线
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 设置坐标轴
    ax.set_xlabel(f'量化比因子($\\lambda$)', fontproperties=zhfont_extra, fontsize=24)
    ax.set_ylabel('相对于逐层量化的改进百分比(%)', fontproperties=zhfont_extra, fontsize=24)
    
    # 设置刻度字体大小
    ax.tick_params(axis='both', labelsize=20)
    
    # 添加图例
    ax.legend(loc='best', prop=zhfont_large, fontsize=20)
    
    # 标题
    plt.title(f'{net_name} - 逐块量化相对逐层量化的性能改进', fontproperties=zhfont_extra, fontsize=24, pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'my_tests/{net_name}_relative_improvement.png', dpi=300, bbox_inches='tight')
    plt.show()

# 示例调用
if __name__ == "__main__":
    # 文件路径
    xlsx_file = 'my_tests/draw_block_adaptive_score.xlsx'
    
    # 模型名称
    # model_name = 'llama-7b'
    model_name = 'Llama-2-7b'
    
    # # 单指标对比
    # compare_block_layer_metrics(xlsx_file, model_name, 'wikitext2')
    # compare_block_layer_metrics(xlsx_file, model_name, 'c4')
    # compare_block_layer_metrics(xlsx_file, model_name, 'average_acc')
    
    # 综合指标对比
    compare_comprehensive_metrics(xlsx_file, model_name)
    
    # 相对改进对比
    plot_relative_improvement(xlsx_file, model_name) 

# pip install openpyxl