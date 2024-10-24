#!/bin/bash

# for seed in 42
# do
# CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --master_port='29500' \
#         --use_env main.py \
#         imr_hideprompt_5e \
#         --model vit_base_patch16_224_dino \
#         --original_model vit_base_patch16_224_dino \
#         --batch-size 24 \
#         --epochs 20 \
#         --data-path ./datasets \
#         --lr 0.0005 \
#         --ca_lr 0.005 \
#         --crct_epochs 30 \
#         --seed $seed \
#         --train_inference_task_only \
#         --output_dir ./output/imr_sup21k_vit_multi_centroid_mlp_2_seed$seed
# done

# Set variables
reg_sub=0.015
reg_glob=0.002

OT=0
delta=100
eval_trick=0
eta=0.02
delta2=15
port='29605'


# Ensure the output directory exists
mkdir -p "output/output_all"

# Correct the output file path
# output_file="./output/output_all/imr_vit_pe_seed${seed}-regsub${reg_sub}-regglob${reg_glob}-OT${OT}-delta${delta}.txt"
output_file="./output/output_all/Eval_imr_vit_pe_seed${seed}-regsub${reg_sub}-regglob${reg_glob}-OT${OT}-delta${delta}_eval-trick${eval_trick}_eta${eta}_eta0${eta_0}_delta2${delta2}.txt"


{
    for seed in 42
    do
        CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port=$port \
            --use_env main.py \
            imr_hideprompt_5e \
            --model vit_base_patch16_224 \
            --original_model vit_base_patch16_224 \
            --batch-size 24 \
            --epochs 150 \
            --data-path ./datasets \
            --crct_epochs 30 \
            --sched cosine \
            --seed $seed \
            --reg_sub $reg_sub \
            --reg_glob $reg_glob \
            --order 1 \
            --length 20 \
            --larger_prompt_lr \
            --trained_original_model ./output/imr_sup21k_vit_multi_centroid_mlp_2_seed$seed \
            --output_dir ./output/imr_dino_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}-OT${OT}-delta${delta} \
            --OT_trick $OT \
            --delta $delta \
            --eval \
            --eval_trick $eval_trick \
            --eta $eta \
            --eta_0 1 \
            --delta2 $delta2 

    done
} > "$output_file" 2>&1

echo "Output has been saved to $output_file"