#!/bin/bash

# # for seed in 42
# # do
# # CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
# #         --nproc_per_node=1 \
# #         --master_port='29500' \
# #         --use_env main.py \
# #         imr_hideprompt_5e \
# #         --model vit_base_patch16_224 \
# #         --original_model vit_base_patch16_224 \
# #         --batch-size 24 \
# #         --epochs 20 \
# #         --data-path ./datasets \
# #         --lr 0.0005 \
# #         --ca_lr 0.005 \
# #         --crct_epochs 30 \
# #         --seed $seed \
# #         --train_inference_task_only \
# #         --output_dir ./output/imr_vit_multi_centroid_mlp_2_seed$seed
# # done

# Set variables
reg=0.8
reg_sub=0.1
reg_glob=0.0
prompt_momentum=0.00001
lr=0.03
ca_lr=0.05
port='29604'


# Ensure the output directory exists
mkdir -p "output/output_all"

# Correct the output file path
output_file="./output/output_all/imr_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}.txt"

{
    for seed in 42
    do
        CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port=$port \
            --use_env main.py \
            imr_hideprompt_5e \
            --model vit_base_patch16_224 \
            --original_model vit_base_patch16_224 \
            --batch-size 24 \
            --epochs 150 \
            --data-path ./datasets \
            --lr $lr \
            --ca_lr $ca_lr \
            --crct_epochs 30 \
            --sched cosine \
            --seed $seed \
            --prompt_momentum $prompt_momentum \
            --reg $reg \
            --reg_sub $reg_sub \
            --reg_glob $reg_glob \
            --order 1 \
            --length 20 \
            --larger_prompt_lr \
            --trained_original_model ./output/imr_vit_multi_centroid_mlp_2_seed$seed \
            --output_dir ./output/imr_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}
    done
} > "$output_file" 2>&1

echo "Output has been saved to $output_file"