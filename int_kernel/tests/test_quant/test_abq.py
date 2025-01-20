import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
from int_kernel.nn.linear import W8A8O16Linear
import safetensors.torch

model = "/workspace/volume/yangzhe/ABQ-LLM/algorithm/quant/Llama-2-7b-chat-hf-w4a4-117/save_dir"

with open('/workspace/volume/yangzhe/ABQ-LLM/algorithm/cache/inps.pt', 'rb') as f:
    inputs = torch.load(f).half()
inputs = inputs.to('cuda:0')

# model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# # # print(model.device)

# q_proj = model.model.layers[0].self_attn.q_proj
# print(q_proj.in_features, q_proj.out_features, q_proj.bias is not None) # 4096 4096 False

# with open('/workspace/volume/yangzhe/ABQ-LLM/int_kernel/tests/test_quant/q_proj.pt', 'wb') as f:
#     torch.save((q_proj.weight, q_proj.bias), f)

# q_proj = nn.Linear(4096, 4096, bias=True).cuda().half()

# model = safetensors.torch.load_file('/workspace/volume/yangzhe/ABQ-LLM/algorithm/quant/Llama-2-7b-chat-hf-w4a4-117/save_dir/model-00001-of-00003.safetensors')

# with torch.no_grad():
#     q_proj.weight.requires_grad_(False)
#     q_proj.bias.requires_grad_(False)
#     q_proj.weight.copy_(model['model.layers.0.self_attn.q_proj.weight'])
#     q_proj.bias.copy_(model['model.layers.0.self_attn.q_proj.bias'])
#     # register 
#     q_proj.zeros = model['model.layers.0.self_attn.q_proj.weight_quantizer.zeros']
#     q_proj.scales = model['model.layers.0.self_attn.q_proj.weight_quantizer.scales']

# with open('/workspace/volume/yangzhe/ABQ-LLM/int_kernel/tests/test_quant/q_abq.pt', 'wb') as f:
#     torch.save(q_proj, f)
# sys.exit()


with open('/workspace/volume/yangzhe/ABQ-LLM/int_kernel/tests/test_quant/q_abq.pt', 'rb') as f:
    q_proj = torch.load(f)
q_proj.scales = q_proj.scales.to('cuda:0')
q_proj.zeros = q_proj.zeros.to('cuda:0')

print(q_proj.weight.shape, q_proj.weight.device) # torch.Size([4096, 4096])
print(q_proj.scales.shape, q_proj.scales.device) # torch.Size([4096, 1])
print(q_proj.zeros.shape, q_proj.zeros.device) # torch.Size([4096, 1])

q_linear = W8A8O16Linear(q_proj.in_features, q_proj.out_features, bias=q_proj.bias is not None).cuda().half()
with torch.no_grad():
    q_linear.weight.copy_(q_proj.weight)
    if q_proj.bias is not None:
        q_linear.bias.copy_(q_proj.bias)
    q_linear.zeros = q_proj.zeros
    q_linear.scales = q_proj.scales


'''
mse_loss = []
mse_loss_absmax = []
with torch.no_grad():
    for i in range(inputs.shape[0]):
        out = q_proj(inputs[i])
        out_q = q_linear(inputs[i])
        out_q_absmax = q_linear.forward_absmax(inputs[i])
        mse_loss.append(torch.mean((out - out_q) ** 2))
        mse_loss_absmax.append(torch.mean((out - out_q_absmax) ** 2))
print(torch.mean(torch.tensor(mse_loss)))           # 0.6587
print(torch.mean(torch.tensor(mse_loss_absmax)))    # 5.5432e-06
'''

mse_loss = []
with torch.no_grad():
    for i in range(inputs.shape[0]):
        out = q_proj(inputs[i])
        out_q = q_linear.forward_abq(inputs[i])
        mse_loss.append(torch.mean((out - out_q) ** 2))
print(torch.mean(torch.tensor(mse_loss)))           # 0.6587
