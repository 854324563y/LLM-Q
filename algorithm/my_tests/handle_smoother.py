import numpy as np
import scipy.optimize as opt
# 测试了一组数据，理想中是单调递增、导函数递减但始终大于0的分布。但实际并不是这样。
# 调整
def smooth_sequence(data):
    n = len(data)
    total_distance = data[-1] - data[0]
    
    def generate_sequence(decay_rate):
        # 生成一个指数衰减的导数序列
        t = np.linspace(0, 1, n-1)
        derivatives = np.exp(-decay_rate * t)
        # 归一化导数使得总和等于总距离
        derivatives = derivatives / np.sum(derivatives) * total_distance
        
        # 通过累积和构建序列
        sequence = [data[0]]
        for d in derivatives:
            sequence.append(sequence[-1] + d)
        return sequence, derivatives
    
    # 通过二分搜索找到最佳的decay_rate
    left, right = 0.1, 5.0
    best_sequence = None
    best_error = float('inf')
    
    for _ in range(20):  # 二分搜索20次
        decay_rate = (left + right) / 2
        sequence, derivatives = generate_sequence(decay_rate)
        
        # 计算与目标点的误差
        error = sum((x - y) ** 2 for x, y in zip(sequence, data))
        
        if error < best_error:
            best_error = error
            best_sequence = sequence
            
        # 根据导数的递减程度调整decay_rate
        if np.any(np.diff(derivatives) > 0):
            left = decay_rate  # 需要更强的衰减
        else:
            right = decay_rate  # 可以尝试更弱的衰减
    
    return best_sequence

# 原始数据
d = "62.25	63.94	64.04 	63.77	64.01	63.77	62.56	62.77"
data = [float(x) for x in d.split()]
adjusted_data = smooth_sequence(data)
if adjusted_data:
    # 只保留小数点两位
    adjusted_data = [round(x, 2) for x in adjusted_data]
    print("\nFinal result:", adjusted_data)
    print("Final differences:", np.diff(adjusted_data))
    # 打印每个差值相对于前一个差值的比例
    diffs = np.diff(adjusted_data)
    ratios = diffs[1:] / diffs[:-1]
    print("Consecutive difference ratios:", [round(r, 3) for r in ratios])
