import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

attn = torch.load('/workspace/volume/yangzhe/ABQ-LLM/algorithm/attn_out_0.pt')
print(attn, attn.shape)

sub_attn = attn[0, :32, :32]

with torch.no_grad():
    mask = torch.triu(torch.ones_like(sub_attn), diagonal=1).bool()
    sub_attn.masked_fill_(mask, float('nan'))

sub_attn_np = sub_attn.detach().cpu().numpy()

plt.figure(figsize=(8, 8))
plt.imshow(sub_attn_np, cmap='Reds', interpolation='nearest', aspect='equal')
# plt.imshow(sub_attn_np, cmap='viridis', interpolation='nearest', aspect='equal')
plt.colorbar(label='Attention Score')
plt.title('Attention Heatmap')
plt.xlabel('Index')
plt.ylabel('Index')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_时分秒
filename = f'attention_heatmap_{timestamp}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')

plt.show()
