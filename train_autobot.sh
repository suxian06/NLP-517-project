CUDA_VISIBLE_DEVICES=0 fairseq-train /homes/iws/sxian/autoencoder/salmon/processed/ --optimizer adam --lr 1e-3 --clip-norm 0.1  --max-tokens 4000 --warmup-updates 4000 --dropout 0.1 --arch autoencoder_roberta_base --save-dir /homes/iws/sxian/autoencoder/salmon/model/