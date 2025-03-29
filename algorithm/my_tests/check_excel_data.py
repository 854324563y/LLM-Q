import pandas as pd
import numpy as np

# 文件路径
xlsx_file = 'my_tests/draw_block_adaptive_score.xlsx'

# 读取Excel文件中的两个表单
try:
    df_block = pd.read_excel(xlsx_file, sheet_name='block')
    df_layer = pd.read_excel(xlsx_file, sheet_name='layer')
    
    # 打印基本信息
    print("Block表格信息:")
    print(f"形状: {df_block.shape}")
    print(f"列名: {df_block.columns.tolist()}")
    print(f"缺失值数量: {df_block.isna().sum().sum()}")
    print(f"无限值数量: {np.isinf(df_block.select_dtypes(include=['float64', 'int64'])).sum().sum()}")
    
    # 打印每一列的非有限值数量
    print("\n每列非有限值检查 (Block表):")
    for col in df_block.columns:
        if pd.api.types.is_numeric_dtype(df_block[col]):
            na_count = df_block[col].isna().sum()
            inf_count = np.isinf(df_block[col]).sum()
            if na_count > 0 or inf_count > 0:
                print(f"列 '{col}': NaN值 = {na_count}, 无限值 = {inf_count}")
    
    # 打印layer表的相同信息
    print("\nLayer表格信息:")
    print(f"形状: {df_layer.shape}")
    print(f"列名: {df_layer.columns.tolist()}")
    print(f"缺失值数量: {df_layer.isna().sum().sum()}")
    print(f"无限值数量: {np.isinf(df_layer.select_dtypes(include=['float64', 'int64'])).sum().sum()}")
    
    # 打印每一列的非有限值数量
    print("\n每列非有限值检查 (Layer表):")
    for col in df_layer.columns:
        if pd.api.types.is_numeric_dtype(df_layer[col]):
            na_count = df_layer[col].isna().sum()
            inf_count = np.isinf(df_layer[col]).sum()
            if na_count > 0 or inf_count > 0:
                print(f"列 '{col}': NaN值 = {na_count}, 无限值 = {inf_count}")
    
    # 检查factor的唯一值
    print("\nFactor值检查:")
    print(f"Block表中的factor值: {sorted(df_block['factor'].unique())}")
    print(f"Layer表中的factor值: {sorted(df_layer['factor'].unique())}")
    
    # 检查共同的factor值
    common_factors = sorted(set(df_block['factor']).intersection(set(df_layer['factor'])))
    print(f"\n共同的factor值: {common_factors}")
    
    # 检查文本中的问题区域
    print("\n检查文本中的问题区域:")
    
    # 检查相对改进计算
    for factor in common_factors:
        block_row = df_block[df_block['factor'] == factor].iloc[0]
        layer_row = df_layer[df_layer['factor'] == factor].iloc[0]
        
        # 检查分母是否为零或接近零
        if abs(layer_row['wikitext2']) < 1e-10:
            print(f"Factor {factor}: layer wikitext2 接近零 ({layer_row['wikitext2']})")
        if abs(layer_row['c4']) < 1e-10:
            print(f"Factor {factor}: layer c4 接近零 ({layer_row['c4']})")
        if abs(layer_row['average_acc']) < 1e-10:
            print(f"Factor {factor}: layer average_acc 接近零 ({layer_row['average_acc']})")
        if abs(layer_row['bitops']) < 1e-10:
            print(f"Factor {factor}: layer bitops 接近零 ({layer_row['bitops']})")
            
    # 打印一些数据样本
    print("\nBlock表数据样本:")
    print(df_block.head())
    print("\nLayer表数据样本:")
    print(df_layer.head())
    
except Exception as e:
    print(f"读取Excel文件时出错: {str(e)}") 