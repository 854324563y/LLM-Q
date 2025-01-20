import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
from int_kernel.nn.linear import W8A8O16Linear


model = "/workspace/volume/inference-soft-data/AE/llm/models/Llama-2-7b-chat-hf"

with open('/workspace/volume/yangzhe/ABQ-LLM/algorithm/cache/inps.pt', 'rb') as f:
    inputs = torch.load(f).half()
inputs = inputs.to('cuda:0')

# model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# # print(model.device)

# q_proj = model.model.layers[0].self_attn.q_proj
# # print(q_proj.in_features, q_proj.out_features, q_proj.bias is not None) # 4096 4096 False

# with open('/workspace/volume/yangzhe/ABQ-LLM/int_kernel/tests/test_quant/q_proj.pt', 'wb') as f:
#     torch.save((q_proj.weight, q_proj.bias), f)


q_proj = nn.Linear(4096, 4096, bias=True).cuda().half()
with open('/workspace/volume/yangzhe/ABQ-LLM/int_kernel/tests/test_quant/q_proj.pt', 'rb') as f:
    q_proj.weight, q_proj.bias = torch.load(f)


q_linear = W8A8O16Linear(q_proj.in_features, q_proj.out_features, bias=q_proj.bias is not None).cuda().half()
with torch.no_grad():
    q_linear.weight.copy_(q_proj.weight)
    if q_proj.bias is not None:
        q_linear.bias.copy_(q_proj.bias)


mse_loss = []
mse_loss_absmax = []
with torch.no_grad():
    for i in range(inputs.shape[0]):
        out = q_proj(inputs[i])
        out_q = q_linear(inputs[i])
        out_q_absmax = q_linear.forward_absmax(inputs[i])
        mse_loss.append(torch.mean((out - out_q) ** 2))
        mse_loss_absmax.append(torch.mean((out - out_q_absmax) ** 2))

print(torch.mean(torch.tensor(mse_loss)))           # 0.0183
print(torch.mean(torch.tensor(mse_loss_absmax)))    # 2.2054e-06