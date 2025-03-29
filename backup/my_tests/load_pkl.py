import pickle

# with open('log/Llama-2-7b-chat-hf-w4a4/Llama-2-7b-chat-hf_divide_metics.pkl', 'rb') as f:
#     metrics = pickle.load(f)

# print(metrics)

with open('log/Llama-2-7b-chat-hf-w4a4/Llama-2-7b-chat-hf_blocks.pkl', 'rb') as f:
    blocks = pickle.load(f)
print(blocks)

with open('log/Llama-2-7b-chat-hf-w4a4/Llama-2-7b-chat-hf_blocks_temp.pkl', 'wb') as f:
    pickle.dump([(0, 1, 2, 3), (3, 4, 5, 6)], f)

# with open('log/Llama-2-7b-chat-hf-w4a4/layer_cost_Llama-2-7b-chat-hf.pkl', 'rb') as f:
#     layer_cost = pickle.load(f)
# print(layer_cost)

# with open('log/Llama-2-7b-chat-hf-w4a4/hm_info_Llama-2-7b-chat-hf.pkl', 'rb') as f:
#     hm_info = pickle.load(f)
# print(hm_info[0]['hm'])

# # draw hm matrix map using matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# # not hot map
# plt.imshow(hm_info[0]['hm'], cmap='gray', interpolation='nearest')
# plt.colorbar()
# plt.show()
# plt.savefig('hm.png')

# with open('log/Llama-2-7b-chat-hf-w4a4-mpq/quant_result_Llama-2-7b-chat-hf.pkl', 'rb') as f:
#     quant_result = pickle.load(f)
# print(quant_result)
