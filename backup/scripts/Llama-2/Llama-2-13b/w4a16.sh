CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/Llama-2-13b --eval_ppl \
--epochs 20 --output_dir ./log/Llama-2-13b-w4a16 \
--wbits 4 --abits 16 --lwc --let \
--save_dir ./quant/Llama-2-13b-w4a16 \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande