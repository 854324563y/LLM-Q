# path/to/dir
# 子目录命名格式为{net}
# 有日志{net}/log_reload_{factor}.txt
# 日志中包含平均weight位宽和activation位宽信息
# 根据net和factor，输出合理的csv表格结构

import os
import json
import glob
import re
from typing import Dict, Any, Optional, Tuple

def find_log(directory: str, factor: str) -> Optional[str]:
    """查找指定factor的日志文件"""
    log_file = os.path.join(directory, f"log_reload_{factor}.txt")
    if os.path.exists(log_file):
        return log_file
    return None

def extract_bitwidth_from_log(log_file: str) -> Optional[Tuple[float, float]]:
    """从日志文件中提取weight和activation位宽数据"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None
            
            # 从后向前搜索最近的位宽信息
            weight_width = None
            activation_width = None
            
            for line in reversed(lines):
                line = line.strip()
                # 匹配weight位宽
                w_match = re.search(r"平均weight位宽: ([\d.]+) bits", line)
                if w_match and weight_width is None:
                    weight_width = float(w_match.group(1))
                
                # 匹配activation位宽
                a_match = re.search(r"平均activation位宽: ([\d.]+) bits", line)
                if a_match and activation_width is None:
                    activation_width = float(a_match.group(1))
                
                # 如果两个值都找到了，就可以返回了
                if weight_width is not None and activation_width is not None:
                    return (weight_width, activation_width)
                    
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    return None

def main(base_path: str):
    """主函数"""
    # 创建CSV表头
    print("net,factor,weight_width,activation_width")
    
    # 遍历目录
    for dirname in os.listdir(base_path):
        if not os.path.isdir(os.path.join(base_path, dirname)):
            continue
            
        # 获取net名称
        net = dirname
        
        # 遍历该网络下的所有日志文件
        dir_path = os.path.join(base_path, dirname)
        # 获取所有可能的factor
        log_files = glob.glob(os.path.join(dir_path, "log_reload_*.txt"))
        for log_file in log_files:
            # 从文件名提取factor
            factor = re.search(r"log_reload_(.*?)\.txt", os.path.basename(log_file)).group(1)
            
            # 提取位宽数据
            result = extract_bitwidth_from_log(log_file)
            if result is not None:
                weight_width, activation_width = result
                # 输出CSV格式数据
                print(f"{net},{factor},{weight_width:.2f},{activation_width:.2f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python handle_bitops.py <directory_path>")
        sys.exit(1)
    
    main(sys.argv[1])