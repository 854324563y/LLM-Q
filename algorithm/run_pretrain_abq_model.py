# Copyright 2024 ByteDance and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
import sys
import random
import numpy as np
import torch
import time
from datautils import get_loaders
import torch.nn as nn
from tqdm import tqdm

from models.int_llama_layer import QuantLlamaDecoderLayer
from quantize.utils import set_quant_state

from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc 
torch.backends.cudnn.benchmark = True

@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    if "c4" in dataset:
        testenc = testenc
    else:
        testenc = testenc.input_ids

    nsamples = testenc.numel() // model.seqlen
    print('nsamples', nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    model.config.use_cache = use_cache



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.deactive_amp = True

    net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = net.split('-')[0]
        
    ## abq_llm在进行完校准准后已经实现let了，这里无需再设置
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": False,
        "dynamic_method": "per_channel",
        "group_size": args.group_size,
        ## 这里为什么lwc设置False
        "lwc":False,
        "disable_zero_point": False
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    enc = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    model.seqlen=2048
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layers[i] = QuantLlamaDecoderLayer(config, layers[i], args)
        ### 推理的时候在QuantLinear.forward中，use_weight_quant=False，不会量化weight
        ### 因weight之前已经fake_quant了
        ### 推理的时候act_quant=True，会量化activation
        set_quant_state(layers[i], weight_quant=False, act_quant=True)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=args.model,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")

    for dataset in ["wikitext2", "c4" , "ptb"]:
        cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
            print(f"load calibration from {cache_testloader}")
        else:
            dataloader, testloader = get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
            )
            torch.save(testloader, cache_testloader)
        # dataloader, testloader = get_loaders(
        #     dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        # )
        print("Dataset:", dataset)
        llama_eval(model, testloader, DEV, dataset, False)

if __name__ == "__main__":
    print(sys.argv)
    main()

    