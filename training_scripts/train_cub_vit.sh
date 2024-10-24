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

reg=0.08
reg_sub=0.25
reg_glob=0.02

OT=0
delta=100
eval_trick=0
eta=0.02
delta2=50
port='29611'

# Ensure the output directory exists
mkdir -p "output/output_all"

# Correct the output file path
# output_file="./output/output_all/cub_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}-OT${OT}-delta${delta}.txt"
output_file="./output/output_all/Eval_cub_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}-OT${OT}-delta${delta}-eta${eta}_eta0${eta_0}_delta2${delta2}.txt"

# {

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
	--output_dir ./output/cub_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}-OT${OT}-delta${delta} \
	--OT_trick $OT \
	--delta $delta \
	--eval \
	--eval_trick $eval_trick \
	--eta $eta \
	--eta_0 1 \
	--delta2 $delta2 \
done

# } > "$output_file" 2>&1

# echo "Output has been saved to $output_file"