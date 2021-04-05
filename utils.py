import pandas as pd
from collections import defaultdict
import numpy as np
# import bert_helper

def get_embedding(sentence, tokenizer, bert_model):
    if type(sentence) == str:
        a_layers = bert_helper.get_layers(sentence, tokenizer, bert_model)
        _ , a_vec = bert_helper.get_embeddings(a_layers, 1)
        return a_vec
    else: 
        embs, length, _ = bert_helper.get_layers_batch(sentence, tokenizer, bert_model)
        res = bert_helper.get_embeddings_batch(embs, length, method=1)
        return list(map(lambda x: x[1], res))
