import cvxpy as cp
import numpy as np

def adaptive_quantization(G, L, schemes_per_module=3):
    G = (G + G.T) / 2
    # 定义变量
    x = cp.Variable(G.shape[0], boolean=True)
    
    # 目标函数：最小化量化误差
    objective = cp.Minimize(cp.quad_form(x, G))
    
    # 约束条件：每个模块只能选择一个配置
    constraints = []
    for i in range(L):
        constraints.append(cp.sum(x[i*schemes_per_module:(i+1)*schemes_per_module]) == 1)
    
    # 定义并求解优化问题
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # 返回结果
    return x.value

# 示例使用
L = 10  # 假设有10个模块
G = np.random.rand(3*L, 3*L)  # 随机生成量化敏感度矩阵
best_config = adaptive_quantization(G, L)
print("最佳量化配置:", best_config)
