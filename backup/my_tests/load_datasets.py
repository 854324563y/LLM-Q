import torch

d = torch.load('cache/dataloader_Llama_wikitext2_128.cache')
t = torch.load('cache/testloader_Llama_wikitext2_all.cache')
t2 = torch.load('cache/testloader_Llama_c4_all.cache')

# print(len(d)) # 128
# print(d[0][0].shape) # torch.Size([1, 2048])

# print(t.input_ids.shape) # torch.Size([1, 341469])
# print(t2.shape) # torch.Size([1, 524288])