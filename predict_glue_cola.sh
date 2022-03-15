CUDA_VISIBLE_DEVICES=1 python /homes/iws/sxian/autoencoder/autobot/run_glue.py \
  --model_name_or_path /homes/iws/sxian/autoencoder/salmon/sentence_classification/checkpoint-2500/ \
  --pretrained_path /homes/iws/sxian/autoencoder/salmon/sentence_classification/checkpoint-2500/checkpoint_best.pt \
  --output_dir /homes/iws/sxian/autoencoder/salmon/sentence_classification/predict/ \
  --do_predict \
  --task_name cola \
  --data_dir /homes/iws/sxian/autoencoder/autobot/sbert/CoLA/ \
  
  #--num_train_epochs 1 \
  #--seed 42 \
  #--learning_rate  2e-5 \
  #--weight_decay 0.1 \
  #--per_device_train_batch_size 16\

  #--eval_steps 500
  
  