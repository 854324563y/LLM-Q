def adjust_average_proportional(numbers, target):
    """
    按比例调整一组数，使它们的平均值变为 target。

    参数：
        numbers：原始数字列表
        target：目标平均值

    返回：
        调整后的数字列表
    """
    if not numbers:
        return []
    
    current_avg = sum(numbers) / len(numbers)
    
    # 如果当前平均值为 0，则无法按比例调整
    if current_avg == 0:
        raise ValueError("当前平均值为0，无法按比例调整")
    
    # 计算比例因子
    factor = target / current_avg
    
    # 按比例调整每个数字
    adjusted_numbers = [round(x * factor, 2) for x in numbers]
    return adjusted_numbers

# 示例
d = "40.44	69.36	72.72	55.37	77.58	67.4"
numbers = [float(i) for i in d.split()]
target_avg = 62.25

adjusted = adjust_average_proportional(numbers, target_avg)
print("原始数字：", numbers)
print("调整后的数字：", adjusted)
print("新的平均值：", sum(adjusted) / len(adjusted))

