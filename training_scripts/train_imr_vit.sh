#!/bin/bash

for seed in 42
do
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port='29500' \
        --use_env main.py \
        imr_hideprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 20 \
        --data-path ./datasets \
        --lr 0.0005 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed $seed \
        --train_inference_task_only \
        --output_dir ./output/imr_vit_multi_centroid_mlp_2_seed$seed
done

# reg=0.5
# reg_sub=0.1
# reg_glob=0.05
# prompt_momentum=0.01
# lr=0.03
# port='29605'

# for seed in 42
# do
# python -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --master_port=$port \
#         --use_env main.py \
#         imr_hideprompt_5e \
#         --model vit_base_patch16_224 \
#         --original_model vit_base_patch16_224 \
#         --batch-size 24 \
#         --epochs 150 \
#         --data-path ./datasets \
#         --lr $lr \
#         --ca_lr 0.005 \
#         --crct_epochs 30 \
# 	--sched cosine \
#         --seed $seed \
# 	--prompt_momentum $prompt_momentum \
# 	--reg $reg \
# 	--reg_sub $reg_sub \
# 	--reg_glob $reg_glob \
#         --order 1 \
# 	--length 20 \
#         --larger_prompt_lr \
#         --trained_original_model ./output/imr_vit_multi_centroid_mlp_2_seed$seed \
# 	--output_dir ./output/cifar100_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}
# done


