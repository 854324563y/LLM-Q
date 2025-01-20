from quantize.utils_divide import determine_blocks
import pickle

def determine_blocks2(args, quant_errors, layer_similarities, sensitivities):
    """确定block划分
    
    基于量化误差、层间相似度和Hessian敏感度三个指标来划分block。
    使用动态规划找到最优的划分方案。
    
    Args:
        args: 配置参数
        quant_errors: 每层的量化误差列表
        layer_similarities: 层间相似度列表（第一层没有前一层，故首项为1）
        sensitivities: 每层的Hessian敏感度列表
        
    Returns:
        list: block划分列表
    """
    error_threshold = args.error_threshold
    similarity_threshold = args.similarity_threshold 
    sensitivity_threshold = args.sensitivity_threshold
    max_block_size = args.max_block_size

    n_layers = len(quant_errors)
    assert len(layer_similarities) == n_layers
    assert len(sensitivities) == n_layers

    # Initialize dynamic programming table
    dp = [float('inf')] * (n_layers + 1)  # Minimum cost for first i layers
    split = [-1] * (n_layers + 1)         # Stores block boundary
    dp[0] = 0  # No cost for zero layers

    # Compute dynamic programming table
    for i in range(1, n_layers + 1):
        for j in range(max(1, i - max_block_size + 1), i + 1):
            # Consider block (j-1, i-1)
            if j > i:
                continue

            block_quant_error = max(quant_errors[j-1:i])
            block_sensitivity = max(sensitivities[j-1:i])
            block_similarity = min(layer_similarities[j-1:i]) if j-1 < i else 1.0

            # Check thresholds for block validity
            if block_quant_error <= error_threshold and \
               block_sensitivity <= sensitivity_threshold and \
               block_similarity >= similarity_threshold:

                cost = dp[j-1] + 1  # Increment cost for adding a block
                if cost < dp[i]:
                    dp[i] = cost
                    split[i] = j-1

    # Reconstruct block boundaries
    blocks = []
    current = n_layers
    while current > 0:
        start = split[current]
        if start == -1:
            break  # Prevent infinite loop if no valid split is found
        blocks.append((start, current - 1))
        current = start

    blocks.reverse()  # Blocks are reconstructed in reverse order
    return blocks

with open('log/Llama-2-7b-chat-hf-w4a4/Llama-2-7b-chat-hf_divide_metics.pkl', 'rb') as f:
    metrics = pickle.load(f)



class Args:
    error_threshold = 0.2
    similarity_threshold = 0.999
    sensitivity_threshold = 0.01
    max_block_size = 3

args = Args()
quant_errors = metrics[0]
layer_similarities = metrics[1]
layer_similarities[0] = 1
sensitivities = metrics[2]

print('quant_errors: ')
print(quant_errors)
print('layer_similarities: ')
print(layer_similarities)
print('sensitivities: ')
print(sensitivities)

blocks = determine_blocks(args, quant_errors, layer_similarities, sensitivities)
print(blocks)

# python -m my_tests.test_determine_blocks
