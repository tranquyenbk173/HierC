# #!/bin/bash

# for seed in 42
# do
# CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --use_env main.py \
#         cifar100_hideprompt_5e \
#         --original_model vit_base_patch16_224_dino \
#         --model vit_base_patch16_224_dino \
#         --batch-size 24 \
#         --data-path ../Z.Data/ \
#         --output_dir ./output/cifar100_dino_multi_centroid_mlp_2_seed$seed \
#         --epochs 20 \
#         --sched constant \
#         --seed $seed \
#         --train_inference_task_only \
#         --lr 0.0005 
# done

reg_sub=0.01
reg_glob=0.001

OT=0
delta=100
eval_trick=0
eta=0.02 #0 0.01 0.02 0.03 0.1 0.5
delta2=150 #15 50 150 500 1500
port='29524'

# Ensure the output directory exists
mkdir -p "output/output_all"

# Correct the output file path
# output_file="./output/output_all/cifar100_vit_pe_seed${seed}-regsub${reg_sub}-regglob${reg_glob}-OT${OT}-delta${delta}.txt"
output_file="./output/output_all/Eval_cifar100_vit_pe_seed${seed}-regsub${reg_sub}-regglob${reg_glob}-OT${OT}-delta${delta}_eval-trick${eval_trick}_eta${eta}_eta0${eta_0}_delta2${delta2}.txt"

# {

for seed in 422
do
	CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
		--nproc_per_node=1 \
		--master_port=$port \
		--use_env main.py \
		cifar100_hideprompt_5e \
		--model vit_base_patch16_224 \
		--original_model vit_base_patch16_224 \
		--batch-size 24 \
		--epochs 50 \
		--data-path ../Z.Data/ \
		--crct_epochs 30 \
		--seed $seed \
		--reg_sub $reg_sub \
		--reg_glob $reg_glob \
		--order 1 \
		--length 5 \
		--sched step \
		--larger_prompt_lr \
		--trained_original_model ./output/cifar100_sup21k_multi_centroid_mlp_2_seed42 \
		--output_dir ./output/cifar100_vit_pe_seed${seed}-reg${reg}-regsub${reg_sub}-regglob${reg_glob}-prompt_momentum${prompt_momentum}-lr${lr}-calr${ca_lr}-OT${OT}-delta${delta} \
		--OT_trick $OT \
		--delta $delta \
		--eval \
		--eval_trick $eval_trick \
		--eta $eta \
		--eta_0 1 \
		--delta2 $delta2 

done

# } > "$output_file" 2>&1

echo "Output has been saved to $output_file"

