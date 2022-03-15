CUDA_VISIBLE_DEVICES=1 python /homes/iws/sxian/autoencoder/autobot/run_glue.py \
  --do_train \
  --model_name_or_path /homes/iws/sxian/autoencoder/salmon/processed/ \
  --task_name cola \
  --data_dir /homes/iws/sxian/autoencoder/autobot/sbert/CoLA/ \
  --output_dir /homes/iws/sxian/autoencoder/salmon/sentence_classification/cola/ \
  --overwrite_output_dir \
  --num_train_epochs 10 \
  --seed 42 \
  --learning_rate  2e-5 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 16\
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 1 \
  --eval_steps 500
  
  