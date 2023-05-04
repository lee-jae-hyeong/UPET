export TASK_NAME=superglue
export DATASET_NAME=rte
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=5

bs=4
lr=1e-2
dropout=0.1
psl=20
epoch=100

python3 run.py \
  --model_name_or_path /wjn/pre-trained-lm/roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --overwrite_cache \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 42 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --num_examples_per_label 16 \
  --prompt_ptuning \
  --use_pe
 
  # --head_prefix \
  # --use_pe
  # --adapter \
  # --adapter_dim 256 \
  # --adapter_choice list \

