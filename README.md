# NLP-517-project
Final project code for the NLP-517 class at UW.

# Dependency
Python=3.7.11 \
pytorch=1.11.0  \
cudatoolkit=11.3.1 \
fairseq=0.9.0 \
transformer=3.3.1 \
sentence-transformers=0.3.7.2 \
nltk=3.6.7

# Data download instruction

### pretraining
Download the English wiki data
```bash
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```
Then install wikiextractor to extract the English wikidata
```bash
pip install wikiextractor
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2
```
See detail at https://github.com/attardi/wikiextractor


Download the bookcorpus data for pretraining:
```bash
python download_files.py --list url_list.jsonl --out out_txts
```
The downloaded bookcorpus dataset for this project can be found at
```bash
/homes/iws/sxian/autoencoder/salmon/bookcorpus/bookCorpus.txt
```
Which contains 286 books.

### fine-tuning sentence similarity
Download the data for sentence similarity task ('AllNLI.zip', 'stsbenchmark.zip', 'wikipedia-sections-triplets.zip', 'STS2017.en-de.txt.gz', 'TED2013-en-de.txt.gz', 'xnli-en-de.txt.gz')
```bash
python get_data.py
```
### fine-tuning sentence classification
Download data for sentence-classification task, use:
```bash
bash download_glue.sh
```

# Preprocessing command 
The bookcorpus and English wikidata is used for the pretraining process. After extracting the wikidata, only a fraction of them is used for training (folder AA to AP, the UW server limited individual space to be within 25G).
Each text file is tokenized into sentences, using nltk, see the jupyter notebook, the example script is below
```python
# preprocess book corpous!
with open("/homes/iws/sxian/autoencoder/salmon/processed/bookSentences.txt","w") as out:
    for fe in allfiles:
        with open(workdir+fe,"r") as f:
            txt = f.read()
        txt = nltk.sent_tokenize(txt)
        for line in txt:
            line = nltk.word_tokenize(line)
            out.write(" ".join(line)+"\n")
            
# preprocess wikitext!
workdir = "/homes/iws/sxian/autoencoder/wikiextractor/text/"
allfiles = os.listdir(workdir)

with open("/homes/iws/sxian/autoencoder/salmon/processed/wikiSentences.txt","w") as out:
    for _f in allfiles:
        allfiles_subdir = os.listdir(workdir + _f +"/")
        for fe in allfiles_subdir:
            with open(workdir + _f +"/"+ fe,"r") as f:
                txt = f.read()
            txt = nltk.sent_tokenize(txt)
            for line in txt:
                line = nltk.word_tokenize(line)
                out.write(" ".join(line)+"\n")
```
All text file is concatenated into one single file, using the linux command cat.
The singe file is named train.txt, then we splitted the file into train, valid, and test set (see juypyter notebook):
Example code
```python
with open("/homes/iws/sxian/autoencoder/salmon/processed/train.txt","r") as f:
    whole_datasets = f.read()
whole_datasets = whole_datasets.split("\n")
print(f'number of lines {len(whole_datasets)}')

n = 13173826
train_L, valid_L = int(np.ceil(n * .8)), int(np.ceil(n * .1))
test_L = n - train_L - valid_L

### write into train.txt, valid.txt, test.txt
suf = "_L"
cur = 0
for term,size in zip(["train","valid","test"],[train_L, valid_L, test_L]):
    with open("/homes/iws/sxian/autoencoder/salmon/processed/"+ term +".txt","w") as f:
        for line in whole_datasets[cur:size]:
            f.write(line + "\n")
    cur = size
```
# Training code

### preprocess, tokenization and bpe encoding
Preprocess the text file, encoding using autoencoder_roberta_base model, this will output tokenized file as train.tok, valid.tok, test.tok.
```python
bash tokenize_data.sh
```
Encoding the tokenized file (train.tok, valid.tok, and test.tok) as bpe, based on the script ```python spm_train.py```
```python
train.sh
```
### pretraining - AUTOBOT
Pretrain the AUTOBOT model using the preprocessed training data. This will save checkpoint_{epochs}.pt in the /model/ directory. The model is then ready for fine-tuning.
The training hyperparameter follows the description in the paper, detail can be found from the ```python train_autobot.sh``` file.
```python
train_autobot.sh
```
### fine-tuning - sentence similarity
Fine tune the model, preparing for the sentence similarity task. The command below trains the model on the NLI dataset for 1 epoch, including 58880 steps.
```python
train-autobot-sentence-similarity.sh
```
### fine-tuning - sentence classification
Fine tune the model, preparing for the sentence classification task. The command below trains the model on the CoLA dataset for 10 epochs.
```python
train_glue_cola.sh
```
Fine tune the model on the SST-2 dataset. The command below trains the model on the SST-2 dataset for 10 epochs.
```python
train_glue_sst.sh
```

# Evaluation code
The evaluation of pretrained model is not presented from the paper, thus we skipped this step, since no comparison can be made.
We evaluated the model performance on two tasks, sentence similarity and sentence classification.
To evaluate the performance on CoLA, run
```python
eval_glue_cola.sh
```

To evaluate the performance on SST-2, run
```python
eval_glue_sst2.sh
```

# Pretrained model
Not included in this repo, pls find the file under the directory /homes/iws/sxian/autoencoder/salmon/processed/checkpoint_best.pt. The file size is 1.5G.

# Table of results

Results on STS test dataset, sentence similarity task.
|| Pearson correlation | Spearman's correlation |
| --- | --- | --- |
| Cosine similarity | 0.7414 | 0.7717 |
| Manhattan-Distance | 0.7466 | 0.7529 |
| Euclidean-Distance | 0.7470 | 0.7536 | 
| Dot-Product-Similarity | 0.5426 | 0.5399 |

Results on the CoLA and SST-2 dataset, sentence classification task.
|| CoLA @ epoch 6 | CoLA @ epoch 10| SST-2 @ epoch 10|
| Median accuracy |0.636 | 0.624 | 0.938 |
