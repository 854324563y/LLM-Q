import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

zhfont = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=16)  # 默认中文字体大小
zhfont_large = matplotlib.font_manager.FontProperties(fname="/workspace/volume/yangzhe/fonts/SourceHanSansSC-Regular.otf", size=20)  # 大号中文字体

def parse_log_file(log_file):
    # 用于存储每层的loss数据
    layer_losses = defaultdict(list)
    current_layer = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # 检测新的层开始
            layer_start = re.search(r'=== Start quantize layer (\d+) ===', line)
            if layer_start:
                current_layer = int(layer_start.group(1))
                
            # 提取loss信息
            loss_info = re.search(r'layer (\d+) iter (\d+) loss:(\d+\.\d+)', line)
            if loss_info:
                layer = int(loss_info.group(1))
                iter_num = int(loss_info.group(2))
                loss = float(loss_info.group(3))
                layer_losses[layer].append(loss)
    
    return layer_losses

def plot_losses(layer_losses):
    plt.figure(figsize=(12, 8))
    
    for layer, losses in layer_losses.items():
        if layer >= 11:
            continue
        plt.plot(range(len(losses)), losses, label=f'层 {layer}', marker='o')
    
    plt.xlabel('训练轮次', fontproperties=zhfont_large)
    plt.ylabel('损失值', fontproperties=zhfont_large)
    plt.title('Llama-2-13b每轮次的损失值变化', fontproperties=zhfont_large)
    plt.legend(prop=zhfont, loc='upper right')  # 将图例放在右上角，使用中文字体
    plt.grid(True)
    
    # 设置x轴为整数刻度
    max_epochs = max(len(losses) for losses in layer_losses.values())
    plt.xticks(range(max_epochs), fontproperties=zhfont)
    plt.yticks(fontproperties=zhfont)  # 设置y轴刻度字体
    plt.savefig('layer_losses.png')
    plt.close()

def main():
    #log_file = '/workspace/volume/yangzhe/ABQ-LLM/algorithm/log-calibration-compensation-lwc/Llama-2-7b-hf-w4a4/log_rank0_1739952594.txt'
    log_file = '/workspace/volume/yangzhe/ABQ-LLM/algorithm/log-calibration-compensation-lwc/Llama-2-13b-hf-w4a4/log_rank0_1740035200.txt'
    # 解析日志文件
    layer_losses = parse_log_file(log_file)
    # print(layer_losses)
    
    l = []

    # 打印统计信息
    for layer, losses in layer_losses.items():
        print(f"\n层 {layer} 的统计信息:")
        print(f"训练轮数: {len(losses)}")
        print(f"初始 loss: {losses[10]:.6f}")
        print(f"最终 loss: {losses[-1]:.6f}")
        print(f"loss 降低率: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
        l.append((losses[9] - losses[-1]) / losses[9] * 100)
        # print(f"10 loss 降低率: {((losses[9] - losses[-1]) / losses[9] * 100):.2f}%")
    print(sum(l) / len(l))
    
    # 绘制损失曲线
    plot_losses(layer_losses)

if __name__ == "__main__":
    main()