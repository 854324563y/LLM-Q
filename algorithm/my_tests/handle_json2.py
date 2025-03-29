import os
import json
import glob
import re
from typing import Dict, Any, Optional

### 根据日志找到数据集得分

def find_latest_log(directory: str) -> Optional[str]:
    """查找目录下最新的日志文件"""
    pattern = os.path.join(directory, "log_rank0_*.txt")
    files = glob.glob(pattern)
    #print(files)
    if not files:
        return None
    
    # 获取所有日志文件的创建时间
    file_times = []
    for file in files:
        try:
            # 从文件名中提取时间戳
            timestamp = os.path.basename(file).split('_')[-1].replace('.txt', '')
            file_times.append((file, float(timestamp)))
        except (IndexError, ValueError):
            continue
    
    if not file_times:
        return None
    
    # 按时间戳排序并返回最新的文件
    return max(file_times, key=lambda x: x[1])[0]

def extract_json_from_log(log_file: str) -> Optional[Dict[str, Any]]:
    """从日志文件中提取JSON数据"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # 从后向前查找包含JSON数据的行
            for line in reversed(lines):
                if "INFO {" in line:
                    # 提取JSON部分
                    json_str = line.split("INFO ", 1)[1].strip()
                    # 将单引号替换为双引号，将None替换为null
                    json_str = json_str.replace("'", '"').replace("None", "null")
                    ## print(f"Attempting to parse JSON: {json_str}")
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error in {log_file}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    return None

def process_metrics(json_data: Optional[Dict[str, Any]], dataset_name: str = None) -> tuple:
    """处理JSON数据，提取所需指标"""
    if not json_data:
        return 0, 0, {} if dataset_name == "all" else 0

    try:
        wikitext2 = json_data.get('wikitext2', 0)
        c4 = json_data.get('c4', 0)
        
        data = json_data.get('results', {})
        metrics = {
            'arc_challenge': data.get('arc_challenge', {}).get('acc', 0) * 100,
            'arc_easy': data.get('arc_easy', {}).get('acc', 0) * 100,
            'boolq': data.get('boolq', {}).get('acc', 0) * 100,
            'hellaswag': data.get('hellaswag', {}).get('acc', 0) * 100,
            'piqa': data.get('piqa', {}).get('acc', 0) * 100,
            'winogrande': data.get('winogrande', {}).get('acc', 0) * 100
        }
        
        if dataset_name == "all":
            return wikitext2, c4, metrics
        elif dataset_name:
            dataset_score = data.get(dataset_name, {}).get('acc', 0) * 100
            return wikitext2, c4, dataset_score
        else:
            average = sum(metrics.values()) / len(metrics)
            return wikitext2, c4, average
    except Exception as e:
        print(f"Error processing metrics: {e}")
        return 0, 0, {} if dataset_name == "all" else 0

def main(base_path: str, dataset_name: str = None):
    """主函数"""
    # 创建CSV表头
    if dataset_name == "all":
        print("net,factor,wikitext2,c4,arc_challenge,arc_easy,boolq,hellaswag,piqa,winogrande")
    elif dataset_name:
        print(f"net,factor,wikitext2,c4,{dataset_name}")
    else:
        print("net,factor,wikitext2,c4,average_acc")
    
    # 遍历目录
    for dirname in os.listdir(base_path):
        if not os.path.isdir(os.path.join(base_path, dirname)):
            continue
            
        # 解析目录名获取net和factor
        try:
            net, factor = dirname.split('_', 1)
        except ValueError:
            continue
            
        # 获取最新日志文件
        log_file = find_latest_log(os.path.join(base_path, dirname))
        if not log_file:
            continue
            
        # 提取并处理数据
        json_data = extract_json_from_log(log_file)
        wikitext2, c4, result = process_metrics(json_data, dataset_name)
        
        # 输出CSV格式数据
        if dataset_name == "all":
            scores = [f"{result.get(ds, 0):.2f}" for ds in ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'piqa', 'winogrande']]
            print(f"{net},{factor},{wikitext2:.4f},{c4:.4f},{','.join(scores)}")
        else:
            score = result if isinstance(result, (int, float)) else 0
            print(f"{net},{factor},{wikitext2:.4f},{c4:.4f},{score:.2f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) not in [2, 3]:
        print("Usage: python handle_json2.py <directory_path> [datasetname]")
        print("Available datasets: arc_challenge, arc_easy, boolq, hellaswag, piqa, winogrande, all")
        sys.exit(1)
    
    dataset_name = sys.argv[2] if len(sys.argv) == 3 else None
    main(sys.argv[1], dataset_name)

    ## python my_tests/handle_json2.py ./log-adaptive-calibration all