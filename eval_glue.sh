CUDA_VISIBLE_DEVICES=1 python /homes/iws/sxian/autoencoder/autobot/run_glue.py \
  --do_eval \
  --model_name_or_path /homes/iws/sxian/autoencoder/salmon/sentence_classification \
  --output_dir /homes/iws/sxian/autoencoder/salmon/sentence_classification/ \
  --num_train_epochs 1 \
  --seed 42 \
  --learning_rate  2e-5 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 16\
  --task_name cola \
  --data_dir /homes/iws/sxian/autoencoder/autobot/sbert/CoLA/ \
  --eval_steps 500
  
  