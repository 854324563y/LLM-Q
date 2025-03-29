import torch

# model = torch.load('act_scales/Llama-2-7b-chat-hf.pt')
# print(type(model))
# for name, param in model.items():
#     print(name, param.shape)

'''
model.layers.0.self_attn.q_proj torch.Size([4096])
model.layers.0.self_attn.k_proj torch.Size([4096])
model.layers.0.self_attn.v_proj torch.Size([4096])
model.layers.0.self_attn.o_proj torch.Size([4096])
model.layers.0.mlp.gate_proj torch.Size([4096])
model.layers.0.mlp.up_proj torch.Size([4096])
model.layers.0.mlp.down_proj torch.Size([11008])
'''

'''
inps = torch.load('/workspace/volume/yangzhe/ABQ-LLM/algorithm/cache/inps.pt')
print(inps)
print(inps.shape)
'''

attn = torch.load('/workspace/volume/yangzhe/ABQ-LLM/algorithm/attn_out_0.pt')
print(attn)
print(attn.shape)
