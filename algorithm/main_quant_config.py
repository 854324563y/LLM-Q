import os
import sys
import random
import numpy as np
import torch
import time
from datautils import get_loaders
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.blocks.block_wise_quant_config_search import search_quant_config, load_search_quant_config
from quantize.blocks.block_wise_quant_config_search_parallel import search_quant_config_parallel, search_quant_config_parallel2
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
import pickle
from models.LMClass import LMClass

import pdb

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--blocks_pkl", type=str, help="path of blocks pkl")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--reload", action="store_true", help="load search quant config")
    parser.add_argument("--parallel", action="store_true", help="use parallel search")
    parser.add_argument("--parallel2", action="store_true", help="use parallel search with matrix computation parallelization")
    parser.add_argument("--size_bound_factor", type=float, default=1.0, help="size bound")
    parser.add_argument("--bitops_bound_factor", type=float, default=0.7, help="bitops bound")
    args = parser.parse_args()

    args.weight_quant_params = {
        # "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": None,
        "lwc": False,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        # "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }



    # blocks contains tuple of [start_layer, end_layer), the end_layer is exclusive
    # such as [(0, 1), (1, 2, 3)], first block is single layer 0, second block contains two layers 1 and 2
    with open(args.blocks_pkl, 'rb') as f:
        blocks = pickle.load(f)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

        
    args.deactive_amp = True
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]


    if args.reload:
        load_search_quant_config(None, blocks, args, None, None)
        return

    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    # load calibration dataset
    cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        logger.info(f"load calibration from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        torch.save(dataloader, cache_dataloader) 

    logger.info(blocks)


    if args.parallel2:
        search_quant_config_parallel2(lm, blocks, args, dataloader, logger)
    elif args.parallel:
        search_quant_config_parallel(lm, blocks, args, dataloader, logger)
    else:
        search_quant_config(lm, blocks, args, dataloader, logger)

    # with open(f'{args.output_dir}/{args.net}_quant_config.pkl', 'wb') as f:
    #     pickle.dump(quant_config, f)

if __name__ == "__main__":
    print(sys.argv)
    main()
    # pip install cvxpy gurobipy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
    # CUDA_VISIBLE_DEVICES=0 python main_quant_config.py --model /workspace/volume/inference-soft-data/AE/llm/models/Llama-2-7b-chat-hf --output_dir ./log/Llama-2-7b-chat-hf-w4a4-mpq --blocks_pkl log/Llama-2-7b-chat-hf-w4a4/Llama-2-7b-chat-hf_blocks.pkl --nsamples 128

    # python main_quant_config.py --model /workspace/volume/inference-soft-data/AE/llm/models/Llama-2-7b-chat-hf --output_dir ./log/Llama-2-7b-chat-hf-mpq-paral --blocks_pkl log/Llama-2-7b-chat-hf-w4a4/Llama-2-7b-chat-hf_blocks.pkl --nsamples 128 --parallel 2>&1 | tee log/Llama-2-7b-chat-hf-mpq-paral/log.txt

    # bitops_bound_factor = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # size_bound_factor = [0.65, 0.7, 0.75]