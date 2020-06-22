#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=16384 # Memory - Use up to 8G
#SBATCH --time=0 # No time limit
#SBATCH -p gpu
#SBATCH --nodelist=boston-2-31
#SBATCH --gres=gpu:4
#SBATCH --mail-user=hongqiay@andrew.cmu.edu
#SBATCH --mail-type=END

export GLUE_DIR=../glue_data/
export TASK_NAME=MRPC

python -m torch.distributed.launch --nproc_per_node 4 ./examples/text-classification/run_glue.py   \
    --model_name_or_path bert-base-uncased \
    --task_name MRPC \
    --do_train   \
    --do_eval   \
    --data_dir $GLUE_DIR/MRPC/   \
    --max_seq_length 128   \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --output_dir ./glue_output/mrpc_output/ \
    --overwrite_output_dir   \
    --overwrite_cache \
