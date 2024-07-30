CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/Llama-2-13b --eval_ppl \
--epochs 40 --output_dir ./log/Llama-2-13b-w2a6 \
--wbits 2 --abits 6 --lwc --let \
--save_dir ./quant/Llama-2-13b-w2a6 \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande