import pickle

with open('./log-divide/Llama-2-13b-hf-w4a4/Llama-2-13b-hf_blocks.pkl', 'rb') as f:
    blocks = pickle.load(f)
    print(blocks)

    processed_blocks = []
    for start, end in blocks:
        processed_blocks.append(list(range(start, end)))
    blocks = processed_blocks
    print(processed_blocks)
# for block_idx, block in enumerate(blocks):
#     print(block_idx)
#     print(block)