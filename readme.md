# Bioembeddings torchserve quickstart

## Overview
We are going to create a torchserve archive (.mar) in a two-step approach. Feel free to skip a step if you've already got the required files.

## Model preparation
### Step 0. Downloading the model
We need a BERT model in a torch-like format. They can be downloaded from [the official page](https://github.com/google-research/bert). 

Files required after this step:
```
biobert_large/
├── bert_config.json
├── pytorch_model.bin
└── vocab.txt
```

### Step 1. Tracing
Torchserve operates on statically traced models. The most irritating part is that if something goes wrong in this step, the torchserve will not able of detecting it - the only hint will be the fact that the produced output will be mostly random.

Command: 
```
python3 trace_model.py
```

Files required after this step:
```
models
├── biobert_batch.pt
└── vocab.txt
```

### Step 2. Archiving
Now, we can pack all the files we need into a .mar archive for the torchserve.

Command:
```
torch-model-archiver --model-name biobert_batch \--version 1.0 \--serialized-file ./models/biobert_batch.pt \--extra-files ./models/vocab.txt,./utils.py,./bert_helper.py,./handler.py,./bert_tokenizer.py --handler my_handler.py  --export-path ./model-store -f
```
Files required after this step:
```
model-store
└── biobert_batch.mar
```

