CUDA_VISIBLE_DEVICES=1 python /homes/iws/sxian/autoencoder/autobot/run_glue.py \
  --do_train \
  --do_eval \
  --model_name_or_path /homes/iws/sxian/autoencoder/salmon/processed/ \
  --task_name SST-2 \
  --data_dir /homes/iws/sxian/autoencoder/autobot/sbert/SST-2/ \
  --output_dir /homes/iws/sxian/autoencoder/salmon/sentence_classification/sst2/ \
  --overwrite_output_dir \
  --num_train_epochs 10 \
  --seed 42 \
  --learning_rate  2e-5 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 16\
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --evaluate_during_training \
  --greater_is_better \
  --logging_steps 500 \

  