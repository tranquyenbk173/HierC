#!/bin/bash

for seed in 42 
do
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        five_datasets_hideprompt_5e \
        --original_model vit_base_patch16_224 \
        --model vit_base_patch16_224 \
        --batch-size 32 \
        --data-path ../Z.Data \
        --output_dir ./output/5datasets_vit_multi_centroid_mlp_2_seed$seed \
        --epochs 20 \
        --sched constant \
        --seed $seed \
        --train_inference_task_only \
        --lr 0.001 
done

reg=0.01
reg_sub=0.0
reg_glob=0.0
prompt_momentum=0.0001
lr=0.03
ca_lr=0.05
port='29609'

# Ensure the output directory exists
mkdir -p "output/output_all"

# Correct the output file path
output_file="./output/output_all/5dataset_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}.txt"

{

# for seed in 42
# do
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --master_port=$port \
#         --use_env main.py \
#         five_datasets_hideprompt_5e \
#         --original_model vit_base_patch16_224 \
#         --model vit_base_patch16_224 \
#         --batch-size 32 \
#         --data-path ./datasets \
#         --output_dir ./output/5datasets_vit_pe_seed$seed \
#         --epochs 20 \
#         --sched constant \
#         --lr $lr \
#         --ca_lr $ca_lr \
#         --clip-grad 2 \
#         --reg $reg \
#         --reg_sub $reg_sub \
#         --reg_glob $reg_glob \
#         --order 1 \
#         --prompt_momentum $prompt_momentum \
#         --seed $seed \
#         --larger_prompt_lr \
#         --trained_original_model ./output/5datasets_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}
# done
} > "$output_file" 2>&1

echo "Output has been saved to $output_file"