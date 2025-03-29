CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/llama-13b --eval_ppl \
--epochs 40 --output_dir ./log/llama-13b-w2a8 \
--wbits 2 --abits 8 --lwc --let \
--save_dir ./quant/llama-13b-w2a8 \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande