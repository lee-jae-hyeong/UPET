TASK_NAME="glue"
DATASET_NAME="e_cate2"
CUDA_VISIBLE_DEVICES=''
#!export PJRT_DEVICE='TPU'
bs=4
lr=1e-5
student_lr=3e-5
dropout=0.2
psl=128
student_psl=128
tea_train_epoch=2
tea_tune_epoch=1
stu_train_epoch=2
self_train_epoch=2
pe_type='head_prefix'

#!python3 run.py --model_name_or_path xlm-roberta-large --resume_from_checkpoint /content/drive/MyDrive/UPET/checkpoints/self-training/{DATASET_NAME}-roberta/{pe_type}/checkpoint --task_name $TASK_NAME --dataset_name $DATASET_NAME --overwrite_cache --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size $bs --learning_rate $lr --num_train_epochs $tea_train_epoch --pre_seq_len $psl --output_dir /content/drive/MyDrive/UPET/checkpoints/self-training/{DATASET_NAME}-roberta/{pe_type} --overwrite_output_dir --hidden_dropout_prob $dropout --seed 42 --save_strategy epoch --evaluation_strategy epoch --save_total_limit 4 --load_best_model_at_end --metric_for_best_model accuracy --report_to none --num_examples_per_label 16 --$pe_type --use_semi --unlabeled_data_num 4096 --unlabeled_data_batch_size 16 --teacher_training_epoch $tea_train_epoch --teacher_tuning_epoch $tea_tune_epoch --student_training_epoch $stu_train_epoch --self_training_epoch $self_train_epoch --pseudo_sample_num_or_ratio 1024 --student_pre_seq_len $student_psl --post_student_train --save_steps 1 --tpu_num_cores 4
!python3 run.py --model_name_or_path klue/roberta-base --resume_from_checkpoint /content/drive/MyDrive/UPET/checkpoints/self-training/{DATASET_NAME}-bert/{pe_type}/checkpoint --task_name $TASK_NAME --dataset_name $DATASET_NAME --overwrite_cache --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size $bs --learning_rate $lr --num_train_epochs $tea_train_epoch --pre_seq_len $psl --output_dir /content/drive/MyDrive/UPET/checkpoints/self-training/{DATASET_NAME}-roberta-base/{pe_type} --overwrite_output_dir --hidden_dropout_prob $dropout --seed 411 --save_strategy epoch --evaluation_strategy epoch --save_total_limit 2 --load_best_model_at_end --metric_for_best_model accuracy --report_to none --num_examples_per_label 30 --$pe_type --use_semi --unlabeled_data_num 39690 --unlabeled_data_batch_size 16 --teacher_training_epoch $tea_train_epoch --teacher_tuning_epoch $tea_tune_epoch --student_training_epoch $stu_train_epoch --self_training_epoch $self_train_epoch --pseudo_sample_num_or_ratio 9800 --student_pre_seq_len $student_psl --post_student_train --save_steps 1 --cb_loss --cb_loss_beta 0.999 --confidence --prefix_projection --do_predict
