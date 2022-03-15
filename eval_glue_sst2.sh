CUDA_VISIBLE_DEVICES=1 python /homes/iws/sxian/autoencoder/autobot/run_glue.py \
  --model_name_or_path /homes/iws/sxian/autoencoder/salmon/sentence_classification/sst2/ \
  --pretrained_path /homes/iws/sxian/autoencoder/salmon/sentence_classification/sst2/full_model.pt \
  --output_dir /homes/iws/sxian/autoencoder/salmon/sentence_classification/predict/ \
  --do_eval \
  --task_name SST-2 \
  --data_dir /homes/iws/sxian/autoencoder/autobot/sbert/SST-2/ \