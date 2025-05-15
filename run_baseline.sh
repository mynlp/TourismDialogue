#!/bin/bash

code_dir=/home/u00483/repos/TourismDialogue
data_dir=/tmp/code/dataset


function split_dataset() {
    run_split="python split_dataset.py \
    --directory ${data_dir}/tourism_conversation/annotations \
    --output ${data_dir}/tourism_conversation/split_info.json"
    singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir}/container --bind ${code_dir}:/tmp/code ${code_dir}/tourism_project.sif bash -c "cd /tmp/code/dataset && ${run_split}"
}

function create_dataset() {
    mkdir -p ${code_dir}/dataset/proc_data_wz${window_size}
    run_creation="python extract2.py \
    --window-size ${window_size} \
    --split-info ${data_dir}/tourism_conversation/split_info.json \
    --annotation-dir ${data_dir}/tourism_conversation/annotations \
    --output-dir ${data_dir}/proc_data_wz${window_size}_wo_balance"
    singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir}/container --bind ${code_dir}:/tmp/code ${code_dir}/tourism_project.sif bash -c "cd /tmp/code/dataset && ${run_creation}"
}


function train() {
    run_train="python seq2seq2.py"
    singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir}/container --bind ${code_dir}:/tmp/code ${code_dir}/tourism_project.sif bash -c "cd /tmp/code && ${run_train}"
}


function train_seq2seq3() {
    run_train="torchrun \
    --standalone \
    --nproc_per_node=${nproc_per_node} \
    train_baseline.py \
    --train_file ${data_dir}/proc_data_wz${window_size}_wo_balance/train.json \
    --dev_file ${data_dir}/proc_data_wz${window_size}_wo_balance/dev.json \
    --model_name ${model_name} \
    --batch_size ${batch_size} \
    --accumulated_size ${accumulated_size} \
    --lr ${lr} \
    --epochs ${epochs} \
    --optimize_direction min \
    --init_eval_score 100000 \
    --log_dir ${code_dir}/baseline_exps/train-${model_type}-${window_size}-bz$((batch_size * accumulated_size * nproc_per_node))-lr${lr}-ep${epochs} \
    --project_name TravelDialogueBaseline \
    --run_name train-${model_type}-${window_size}-bz$((batch_size * accumulated_size * nproc_per_node))-lr${lr}-ep${epochs}"
    singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir}/container --bind ${code_dir}:/tmp/code ${code_dir}/tourism_project.sif bash -c "cd /tmp/code && ${run_train}"

}


function test_seq2seq3() {
    run_train="torchrun \
    --standalone \
    --nproc_per_node=${nproc_per_node} \
    test_baseline.py \
    --test_file ${data_dir}/proc_data_wz${window_size}_wo_balance/test.json \
    --key_val_file ${code_dir}/value_set2.json \
    --model_name ${model_name} \
    --batch_size 1 \
    --only_load_model \
    ${constraint_decode} \
    --eval_out ${eval_out} \
    --log_dir ${code_dir}/baseline_exps/train-${model_type}-${window_size}-bz$((batch_size * accumulated_size * nproc_per_node))-lr${lr}-ep${epochs} \
    --project_name TravelDialogueBaseline \
    --run_name test-${model_type}-${eval_out}-wz${window_size}-bz$((batch_size * accumulated_size * nproc_per_node))-lr${lr}-ep${epochs}"
    singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir}/container --bind ${code_dir}:/tmp/code ${code_dir}/tourism_project.sif bash -c "cd /tmp/code && ${run_train}"

}
#split_dataset
#create_dataset
#train
model_type=mt5-base
model_name='google/mt5-base'
#model_type=bart-base
#model_name=ku-nlp/bart-base-japanese
#model_type=mt0-base
#model_name=bigscience/mt0-base
#window_size=1
batch_size=64
lr=9e-5
epochs=20
accumulated_size=1
nproc_per_node=1
constraint_decode="--constraint_decode"
eval_out=eval_constraint_decode
#eval_out=eval


#train_seq2seq3
test_seq2seq3

