# 给定/path/to/dir
# 路径下有若干/path/to/dir/{net}文件夹，每个文件夹下有若干文件，读取/path/to/dir/{net}/log_reload_{factor}.txt日志，factor如0.3、0.35等。
# 每个日志文件下记录着自适应量化的量化位宽分配结果，每层有6个线性模块，每个模块分配一种位宽（分配结果用0、1、2表示）
# 读取日志中以列表开始的行，如[1, 0, 0, 0, 0, 0]。这些行按层顺序记录了各层的分配结果。
# 不同{net}的层数不同（32或40）。

# 帮我写一个脚本，以直观的方式记录{net}中每层的每个模块，在不同factor下的选择，我想看看有没有规律

import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import glob

# 定义模块名称
MODULE_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']

# 设置全局字体大小
plt.rcParams['font.size'] = 12  # 基础字体大小
plt.rcParams['axes.titlesize'] = 16  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度标签字体大小

zhfont1 = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=18)  # 默认中文字体大小
zhfont_large = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=20)  # 大号中文字体
zhfont_extra = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=24)
def extract_bit_allocation(log_file):
    """从日志文件中提取位宽分配结果"""
    allocations = []
    with open(log_file, 'r') as f:
        for line in f:
            # 只匹配包含6个数字（0-2）的位宽分配行
            match = re.search(r'\[([0-2], [0-2], [0-2], [0-2], [0-2], [0-2])\]', line)
            if match:
                # 提取数字并转换为整数列表
                bits = [int(bit) for bit in match.group(1).split(', ')]
                allocations.append(bits)
    return allocations

def visualize_allocation(net_dir, output_dir='visualization_results'):
    """可视化一个网络在不同factor下的位宽分配"""
    net_name = os.path.basename(net_dir)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有log文件
    log_files = glob.glob(os.path.join(net_dir, 'log_reload_*.txt'))
    
    if not log_files:
        print(f"未找到{net_dir}下的日志文件")
        return
    
    # 提取所有factor值
    factors = []
    for log_file in log_files:
        match = re.search(r'log_reload_([\d\.]+)\.txt', log_file)
        if match:
            factors.append(float(match.group(1)))
    
    # 按factor排序
    factor_log_pairs = sorted(zip(factors, log_files))
    
    # 提取第一个文件的分配结果，确定层数
    first_allocation = extract_bit_allocation(factor_log_pairs[0][1])
    num_layers = len(first_allocation)
    num_modules = len(first_allocation[0]) if first_allocation else 6  # 默认6个模块
    
    print(first_allocation, num_layers, num_modules)

    # 创建数据结构存储所有factor下的分配结果
    all_allocations = {}
    for factor, log_file in factor_log_pairs:
        allocation = extract_bit_allocation(log_file)
        if len(allocation) == num_layers:  # 确保层数一致
            all_allocations[factor] = allocation
    
    # 创建热力图
    plt.figure(figsize=(15, num_layers * 0.4))
    
    # 为每个模块创建一个子图
    fig, axes = plt.subplots(num_modules, 1, figsize=(12, num_modules * 3), sharex=True)
    
    # 确保axes始终是数组
    if num_modules == 1:
        axes = [axes]
    
    # 自定义颜色映射
    # colors = ['#2166ac', '#92c5de', '#f4a582']  # 蓝色(0)、浅蓝色(1)、红色(2)
    colors = ['#BE3C3D', '#FB8769', '#FCBB9F']
    cmap = ListedColormap(colors)
    
    for module_idx in range(num_modules):
        ax = axes[module_idx]
        
        # 创建数据矩阵
        data = np.zeros((num_layers, len(all_allocations)))
        
        for i, (factor, allocation) in enumerate(sorted(all_allocations.items())):
            for layer_idx in range(num_layers):
                data[layer_idx, i] = allocation[layer_idx][module_idx]
        
        # 绘制热力图
        heatmap = sns.heatmap(data, cmap=cmap, ax=ax, cbar=True, 
                    xticklabels=[f"{f:.2f}" for f in sorted(all_allocations.keys())],
                    yticklabels=range(1, num_layers + 1),
                    vmin=0, vmax=2,
                    linewidths=0.2,  # 添加网格线宽度
                    linecolor='gray')  # 设置网格线颜色
        
        # 修改y轴标签，每4层显示一次
        yticks = np.arange(0, num_layers, 4)  # 每4层取一个标签位置
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(i+1) for i in yticks])
        
        # 修改colorbar的刻度标签
        colorbar = heatmap.collections[0].colorbar
        colorbar.set_ticks([0.33, 1.0, 1.67])  # 将刻度放在每个颜色区间的中间
        colorbar.set_ticklabels(['W4A4', 'W4A8', 'W8A8'])
        colorbar.ax.tick_params(labelsize=16)  # 设置colorbar刻度标签大小
        
        ax.set_title(f'{MODULE_NAMES[module_idx]}', fontproperties=zhfont1)
        ax.set_ylabel('层', fontproperties=zhfont_large)
        # 分别设置x轴和y轴的刻度标签大小
        ax.tick_params(axis='x', labelsize=16)  # x轴保持原来的大小
        ax.tick_params(axis='y', labelsize=16)   # y轴设置更小的字体
    
    axes[-1].set_xlabel(f'量化比因子($\\lambda$)\n{net_name}', fontproperties=zhfont_large)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{net_name}_module_allocation.png'), dpi=300)
    
    # 创建另一种可视化：每层在不同factor下的总体位宽分布
    plt.figure(figsize=(14, 10))
    
    # 计算每层的总位宽
    layer_total_bits = {}
    for factor, allocation in sorted(all_allocations.items()):
        layer_bits = []
        for layer_idx in range(num_layers):
            # 假设0,1,2分别代表2,4,8位
            bit_mapping = {0: 2, 1: 4, 2: 8}
            total_bits = sum(bit_mapping[bit] for bit in allocation[layer_idx])
            layer_bits.append(total_bits)
        layer_total_bits[factor] = layer_bits
    
    # 绘制每层总位宽的热力图
    data = np.zeros((num_layers, len(layer_total_bits)))
    for i, (factor, bits) in enumerate(sorted(layer_total_bits.items())):
        for layer_idx in range(num_layers):
            data[layer_idx, i] = bits[layer_idx]
    
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(data, cmap='YlOrRd', 
                xticklabels=[f"{f:.2f}" for f in sorted(layer_total_bits.keys())],
                yticklabels=range(1, num_layers + 1))
    plt.title(f'{net_name} - 每层总位宽分配', fontproperties=zhfont_large)
    plt.xlabel(r'量化比因子 ($\lambda$)', fontproperties=zhfont_large)
    plt.ylabel('层', fontproperties=zhfont_large)
    # 设置刻度标签大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{net_name}_total_bits.png'), dpi=300)
    
    # 绘制每个factor下的位宽分布统计
    plt.figure(figsize=(15, 8))
    
    # 统计每个factor下0,1,2的分布
    bit_counts = {}
    for factor, allocation in sorted(all_allocations.items()):
        counts = [0, 0, 0]  # 分别统计0,1,2的数量
        for layer in allocation:
            for bit in layer:
                counts[bit] += 1
        bit_counts[factor] = counts
    
    # 转换为百分比
    bit_percentages = {}
    for factor, counts in bit_counts.items():
        total = sum(counts)
        bit_percentages[factor] = [count / total * 100 for count in counts]
    
    # 绘制堆叠柱状图
    factors = sorted(bit_percentages.keys())
    bit0_percentages = [bit_percentages[f][0] for f in factors]
    bit1_percentages = [bit_percentages[f][1] for f in factors]
    bit2_percentages = [bit_percentages[f][2] for f in factors]

    
    # ['#BE3C3D', '#FB8769', '#FCBB9F']
    plt.figure(figsize=(15, 8))
    x_pos = np.arange(len(factors))
    plt.bar(x_pos, bit0_percentages, label='W4A4', color='#BE3C3D')
    plt.bar(x_pos, bit1_percentages, bottom=bit0_percentages, label='W4A8', color='#FB8769')
    plt.bar(x_pos, bit2_percentages, bottom=[i+j for i,j in zip(bit0_percentages, bit1_percentages)], 
            label='W8A8', color='#FCBB9F')
    
    plt.xlabel(r'量化比因子 ($\lambda$)', fontproperties=zhfont_extra)
    plt.ylabel('百分比 (%)', fontproperties=zhfont_extra)
    plt.title(f'{net_name}', fontproperties=zhfont_extra)
    plt.legend(fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 设置横坐标刻度和标签
    plt.xticks(x_pos, [f'{f:.2f}' for f in factors], rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{net_name}_bit_distribution.png'), dpi=300)
    
    print(f"已完成{net_name}的可视化分析")

def main():
    """主函数"""
    # 获取用户输入的目录路径
    # base_dir = input("请输入包含网络文件夹的基础路径 (/path/to/dir): ")
    base_dir = '/workspace/volume/yangzhe/ABQ-LLM/algorithm/log-adaptive'
    
    # 获取所有网络文件夹
    net_dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    
    if not net_dirs:
        print(f"在{base_dir}下未找到网络文件夹")
        return
    
    # 为每个网络生成可视化
    for net_dir in net_dirs:
        print(f"正在处理网络: {os.path.basename(net_dir)}")
        visualize_allocation(net_dir)
    
    print("所有网络的可视化分析已完成")

if __name__ == "__main__":
    main()