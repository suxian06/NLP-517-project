fairseq-preprocess --source-lang en --target-lang en --arch autoencoder_roberta_base \
--trainpref /homes/iws/sxian/autoencoder/salmon/processed/train.tok \
--validpref /homes/iws/sxian/autoencoder/salmon/processed/valid.tok \
--testpref /homes/iws/sxian/autoencoder/salmon/processed/test.tok \
--destdir /homes/iws/sxian/autoencoder/salmon/processed/ --workers 128 --srcdict /homes/iws/sxian/autoencoder/salmon/processed/dictionary.json --tgtdict /homes/iws/sxian/autoencoder/salmon/processed/dictionary.json