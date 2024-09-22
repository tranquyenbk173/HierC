#!/bin/bash

# for seed in 42
# do
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --master_port='29700' \
#         --use_env main.py \
#         cub_hideprompt_5e \
#         --model vit_base_patch16_224 \
#         --original_model vit_base_patch16_224 \
#         --batch-size 24 \
#         --epochs 20 \
#         --data-path ../Z.Data/ \
#         --lr 0.01 \
#         --ca_lr 0.005 \
#         --crct_epochs 30 \
#         --seed $seed \
#         --train_inference_task_only \
#         --output_dir ./output/cub_vit_multi_centroid_mlp_2_seed$seed 
# done

reg=0.2
reg_sub=0.5
reg_glob=0.0
prompt_momentum=0.0001
lr=0.03
ca_lr=0.005
port='29608'

# Ensure the output directory exists
mkdir -p "output/output_all"

# Correct the output file path
output_file="./output/output_all/cub_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}.txt"

{

for seed in 42
do
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port=$port \
	--use_env main.py \
	cub_hideprompt_5e \
	--model vit_base_patch16_224 \
	--original_model vit_base_patch16_224 \
	--batch-size 24 \
	--epochs 50 \
	--data-path ../Z.Data/ \
    --lr $lr \
	--ca_lr $ca_lr \
	--crct_epochs 30 \
	--seed $seed \
	--prompt_momentum $prompt_momentum \
	--reg $reg \
	--reg_sub $reg_sub \
	--reg_glob $reg_glob \
    --order 1 \
	--length 20 \
	--trained_original_model ./output/cub_vit_multi_centroid_mlp_2_seed$seed \
	--output_dir ./output/cub_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}
done

} > "$output_file" 2>&1

echo "Output has been saved to $output_file"