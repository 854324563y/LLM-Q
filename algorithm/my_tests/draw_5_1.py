# 给定txt文件，格式如下，记录各层的结果
# '''
# Layer 0 Results:
# Look-ahead layers: 0
# End-to-End MSE: 0.072992
# Cosine Similarity: 0.995406
# --------------------------------------------------

# Layer 1 Results:
# ......
# '''
# 需要使用matplotlib画出各层MSE的折线图


# 给定/path/to/dir/，针对不同的look_ahead_num，有若干日志文件：
# /path/to/dir/{net}-{look_ahead_num}/layer_mse_results_look_ahead_{look_ahead_num}.txt

# 需要画出不同look_ahead_num下的MSE折线图。因不同look_ahead_num的文件中，层数不同，需要将不同look_ahead_num的文件中的层数对齐。


import matplotlib.pyplot as plt
import matplotlib
import re
import os
import glob
from typing import Dict, List, Tuple, Optional

zhfont1 = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf")
zhfont_large = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=20)

def parse_results_file(file_path: str) -> Tuple[List[int], List[float]]:
    """解析结果文件，提取每层的MSE值"""
    layers = []
    mse_values = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # 使用正则表达式匹配每层的结果
    pattern = r'Layer (\d+) Results:.*?End-to-End MSE: ([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for layer, mse in matches:
        layers.append(int(layer))
        mse_values.append(float(mse))
    
    # print(file_path, mse_values)
    
    return layers, mse_values

def align_layer_data(all_results: Dict[int, Tuple[List[int], List[float]]], max_layer: Optional[int] = None) -> Dict[int, Dict[int, float]]:
    """对齐不同look_ahead_num的层数数据
    
    Args:
        all_results: 所有look_ahead_num的结果数据
        max_layer: 可选，手动指定最大层数。如果不指定，则使用所有数据中最小的最大层数
    """
    aligned_data = {}
    
    # 找出所有数据中最小的最大层数
    if max_layer is None:
        max_layer = min(max(layers) for layers, _ in all_results.values())
    
    # 对每个look_ahead_num，创建层数到MSE的映射，只保留到最大层数
    for look_ahead_num, (layers, mse_values) in all_results.items():
        layer_to_mse = dict(zip(layers, mse_values))
        aligned_data[look_ahead_num] = {
            layer: layer_to_mse.get(layer, None) 
            for layer in range(min(layers), max_layer + 1)
        }
    
    return aligned_data

def plot_mse_curves(aligned_data: Dict[int, Dict[int, float]], output_path: str = 'mse_curves.png'):
    """绘制多条MSE折线图"""
    plt.figure(figsize=(12, 8))
    
    # 设置不同的颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for idx, (look_ahead_num, layer_data) in enumerate(sorted(aligned_data.items())):
        layers = []
        mse_values = []
        for layer, mse in sorted(layer_data.items()):
            if mse is not None:  # 只绘制有数据的点
                layers.append(layer)
                mse_values.append(mse)
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(layers, mse_values, 
                marker=marker, 
                linestyle='-', 
                linewidth=2, 
                markersize=8,
                label=f'跨层优化（步长{look_ahead_num}）' if look_ahead_num != 0 else '逐层优化',
                color=color)
        
        # # 在数据点上标注MSE值
        # for x, y in zip(layers, mse_values):
        #     plt.annotate(f'{y:.6f}', 
        #                 (x, y),
        #                 textcoords="offset points",
        #                 xytext=(0, 10),
        #                 ha='center',
        #                 fontsize=10,
        #                 fontproperties=zhfont1)
    
    plt.title('不同优化方式下各层的端到端MSE对比', fontproperties=zhfont1, fontsize=24)
    plt.xlabel('层', fontsize=20, fontproperties=zhfont1)
    plt.ylabel('端到端MSE', fontsize=20, fontproperties=zhfont1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(prop=zhfont_large)
    
    # 设置刻度字体
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_look_ahead_num(file_path: str) -> int:
    """从文件路径中提取look_ahead_num"""
    match = re.search(r'look_ahead_(\d+)\.txt$', file_path)
    return int(match.group(1)) if match else -1

def main():
    # 基础目录路径
    base_dir = '/workspace/volume/yangzhe/ABQ-LLM/algorithm/log-adaptive-calibration-5-1'
    model_name = 'Llama-2-7b-hf-w4a4'
    
    # 可选：手动设置最大层数
    max_layer = 5  # 设置为None则自动使用所有数据中最小的最大层数，或设置具体数字如 max_layer = 10
    
    # 收集所有结果文件
    all_results = {}
    pattern = os.path.join(base_dir, f'{model_name}-*', f'layer_mse_results_look_ahead_*.txt')
    
    for file_path in glob.glob(pattern):
        look_ahead_num = get_look_ahead_num(file_path)
        if look_ahead_num >= 0:
            try:
                layers, mse_values = parse_results_file(file_path)
                all_results[look_ahead_num] = (layers, mse_values)
            except Exception as e:
                print(f"处理文件 {file_path} 时发生错误: {str(e)}")
    
    if not all_results:
        print("未找到任何结果文件！")
        return
    
    # 打印每个look_ahead_num的最大层数
    print("各look_ahead_num的最大层数：")
    for look_ahead_num, (layers, _) in sorted(all_results.items()):
        print(f"Look-ahead {look_ahead_num}: 最大层数 = {max(layers)}")
    
    # 对齐数据并绘图
    aligned_data = align_layer_data(all_results, max_layer)
    output_file = 'mse_curves_comparison.png'
    plot_mse_curves(aligned_data, output_file)
    print(f"\n比较图表已保存为: {output_file}")
    if max_layer is not None:
        print(f"图表显示到第 {max_layer} 层")
    else:
        actual_max = min(max(layers) for layers, _ in all_results.values())
        print(f"图表显示到第 {actual_max} 层（自动选择最小的最大层数）")
    
    # 计算并打印MSE下降比例
    print("\nMSE下降比例分析：")
    print("-" * 50)
    baseline_data = aligned_data.get(0)  # look_ahead_num = 0 的数据作为基准
    if baseline_data:
        for look_ahead_num, layer_data in sorted(aligned_data.items()):
            if look_ahead_num == 0:
                continue
            print(f"\n跨层优化（步长{look_ahead_num}）相比逐层优化的MSE下降比例：")
            for layer, mse in sorted(layer_data.items()):
                if mse is not None and baseline_data.get(layer) is not None:
                    baseline_mse = baseline_data[layer]
                    reduction = (baseline_mse - mse) / baseline_mse * 100
                    print(f"第{layer}层: {reduction:.2f}% (基准MSE: {baseline_mse:.6f}, 优化后MSE: {mse:.6f})")
    else:
        print("未找到逐层优化（look_ahead_num=0）的基准数据，无法计算下降比例。")

if __name__ == '__main__':
    main()



