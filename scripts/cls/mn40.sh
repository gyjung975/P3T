#!/bin/bash

gpu=$1
export CUDA_VISIBLE_DEVICES=${gpu}

main_program=main.py
proj_name=cls
task=cls

model=ULIP_PointBERT
context_len=32
class_pos=middle

dataset=modelnet40
npoints=1024

optim=adamw
lr=0.003
smooth=0.2

gpu=0
epochs=250
batch_size=32
print_freq=20

prompt_encoder=d
offset_generator=dl1
prompt_ratio=0.5
w_proto=5
w_size=0.5
w_center=0.1
w_con=0.5
con_type=cos

exp_name=${task}-mn40-${context_len}v-${class_pos}-\
${prompt_encoder}-${offset_generator}-${prompt_ratio}-\
${w_proto}-${w_size}-${w_center}-${w_con}

output_dir=outputs
log_file=run.log

if [ ! -d ${output_dir}/${proj_name}/${exp_name} ]
then
    mkdir -p ${output_dir}/${proj_name}/${exp_name}
fi

if [ ! -f ${output_dir}/${proj_name}/${exp_name}/${log_file} ]
then
    touch ${output_dir}/${proj_name}/${exp_name}/${log_file}
fi

python ${main_program} --proj_name ${proj_name} --main_program ${main_program} --task ${task} \
    --model ${model} --dataset_name ${dataset} --npoints ${npoints} \
    --optim ${optim} --lr ${lr} --label_smoothing ${smooth} --epochs ${epochs} --batch_size ${batch_size} \
    --exp_name ${exp_name} --output_dir ${output_dir} --print_freq ${print_freq} --gpu ${gpu} \
    --num_learnable_prompt_tokens ${context_len} --class_name_position ${class_pos} \
    --prompt_encoder ${prompt_encoder} --offset_generator ${offset_generator} \
    --prompt_ratio ${prompt_ratio} --con_type ${con_type} \
    --w_proto ${w_proto} --w_size ${w_size} --w_center ${w_center} --w_con ${w_con} \
    --wandb \
    --ulip2 \
    2>&1 | tee ${output_dir}/${proj_name}/${exp_name}/${log_file}
