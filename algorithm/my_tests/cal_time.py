import sys
from datetime import datetime, timedelta

def time_difference(time1, time2):
    # 将输入的时间字符串转换为datetime对象
    format_str = "%H:%M:%S"
    time1_obj = datetime.strptime(time1, format_str)
    time2_obj = datetime.strptime(time2, format_str)
    
    # 计算时间差
    time_diff = time2_obj - time1_obj
    
    # 如果时间1大于时间2，时间差为负，需要调整
    if time_diff.total_seconds() < 0:
        time_diff = time2_obj + timedelta(days=1) - time1_obj
    
    # 计算小时和分钟
    hours = time_diff.seconds // 3600
    minutes = (time_diff.seconds % 3600) // 60
    
    return f"{hours}h{minutes}min"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python time.py time1 time2")
        print("示例：python time.py 04:44:58 10:42:10")
        sys.exit(1)
    
    time1 = sys.argv[1]
    time2 = sys.argv[2]
    
    # 调用函数并输出结果
    result = time_difference(time1, time2)
    print(result)
