CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMA/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log/Llama-2-7b-w4a4 \
--wbits 4 --abits 4 --lwc --let \
--save_dir ./quant/Llama-2-7b-w4a4 \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

# W4A4
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /workspace/volume/inference-soft-data/AE/llm/models/Llama-2-7b-chat-hf  \
--epochs 20 --output_dir ./log/Llama-2-7b-chat-hf-w4a4 \
--save_dir ./quant/Llama-2-7b-chat-hf-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

# resume
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /workspace/volume/inference-soft-data/AE/llm/models/Llama-2-7b-chat-hf  \
--epochs 0 --output_dir ./log/Llama-2-7b-chat-hf-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let \
--resume ./log/Llama-2-7b-chat-hf-w4a4/abq_parameters.pth \
--tasks hellaswag,winogrande
# --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
