# Bioembeddings torchserve quickstart

## Overview
We are going to create a torchserve archive (.mar) in a two-step approach and then use it for setting up a Torchserve instance. Feel free to skip a step if you've already got the required files.

## Requirements
 - Python >= 3.7 with packages from requirements.txt
    - torch-model-archiver is optional
 - Docker

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
Now, we can pack all the files we need into a .mar archive for the torchserve. We can do it either via a docker image or a python package - the choice is yours.

Command:
```bash
docker run -v $(pwd):/app wwydmanski/torch-model-archiver --model-name biobert_batch \--version 1.0 \--serialized-file ./models/biobert_batch.pt \--extra-files ./models/vocab.txt,./utils.py,./bert_helper.py,./handler.py,./bert_tokenizer.py --handler my_handler.py  --export-path ./model-store -f
```
OR, without docker:
```bash
torch-model-archiver --model-name biobert_batch \--version 1.0 \--serialized-file ./models/biobert_batch.pt \--extra-files ./models/vocab.txt,./utils.py,./bert_helper.py,./handler.py,./bert_tokenizer.py --handler my_handler.py  --export-path ./model-store -f
```

Files required after this step:
```
model-store
└── biobert_batch.mar
```

## Running the model
Command:
```bash
docker run --rm -it -p 8080:8080 -v $(pwd)/model-store:/home/model-server/model-store pytorch/torchserve torchserve --start --model-store model-store --models biobert=biobert_batch.mar
```

### Embedding sentences
```
curl --location --request GET 'localhost:8080/predictions/biobert' \
--form 'data="test"'
```

### Checking if everything's working:
```bash
~ $ curl localhost:8080/ping
```
Response:
```
{
  "status": "Healthy"
}
```
