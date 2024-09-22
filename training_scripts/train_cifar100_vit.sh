#!/bin/bash

# for seed in 42
# do
# python -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --use_env main.py \
#         cifar100_hideprompt_5e \
#         --original_model vit_base_patch16_224 \
#         --model vit_base_patch16_224 \
#         --batch-size 24 \
#         --data-path ../Z.Data/ \
#         --output_dir ./output/cifar100_sup21k_multi_centroid_mlp_2_seed$seed \
#         --epochs 20 \
#         --sched constant \
#         --seed $seed \
#         --train_inference_task_only \
#         --lr 0.0005 
# done

reg=0.01
reg_sub=0.1
reg_glob=0.05
prompt_momentum=0.0001
lr=0.03
ca_lr=0.05
port='29503'

# Ensure the output directory exists
mkdir -p "output/output_all"

# Correct the output file path
output_file="./output/output_all/cifar100_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}.txt"

{

for seed in 422
do
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port=$port \
	--use_env main.py \
	cifar100_hideprompt_5e \
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
	--length 5 \
	--sched step \
	--larger_prompt_lr \
	--trained_original_model ./output/cifar100_sup21k_multi_centroid_mlp_2_seed42 \
	--output_dir ./output/cifar100_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}
done

} > "$output_file" 2>&1

echo "Output has been saved to $output_file"