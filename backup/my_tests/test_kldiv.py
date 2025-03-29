from quantize.blocks.block_wise_quant_config_search import kldiv, mse
import torch

# a = torch.randn(10, 10)
# b = torch.randn(10, 10)

# print(kldiv(a, b))
# print(mse(a, b))




# 如果结果合理（例如量化误差较小时 KL 散度接近 0），代码实现是正确的
logits_fp = torch.randn(4, 10)  # Full-precision logits
logits_quant = logits_fp + torch.randn(4, 10) * 0.1  # Simulate quantized logits

kl_value = kldiv(logits_quant, logits_fp)
print("KL Divergence:", kl_value)



# 假设 quant_out 和 out 是模型的输出 logits
quant_out = torch.randn(2, 2048, 4096)  # 形状为 (batch_size=2, seq_len=5, vocab_size=10)
out = torch.randn(2, 2048, 4096)

# 计算 KL 散度
kl_divergence = kldiv(quant_out, out)

print(f"KL Divergence: {kl_divergence}")

