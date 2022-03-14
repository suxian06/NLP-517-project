CUDA_VISIBLE_DEVICES=1 python /homes/iws/sxian/autoencoder/autobot/sbert/sbert_nli.py \
  --model_path /homes/iws/sxian/autoencoder/salmon/processed/ \
  --save_path /homes/iws/sxian/autoencoder/salmon/sentence_similarity/ \
  --epochs 1 \
  --seed 42 \
  --lr 2e-5 \
  --weight_decay 0.1 \
  --valid_freq 500 \
  --batch_size 16