import torch

def get_layers(sentence, tokenizer, model):
    if tokenizer is None:
        indexed_tokens = sentence
    else:
        marked_text = "[CLS] " + sentence + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    if len(indexed_tokens)>512:
        indexed_tokens = indexed_tokens[-512:]
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    return encoded_layers

def get_layers_batch(sentence_batch, tokenizer, model):
    tokens_tensor = []
    segments_tensor = []
    for sentence in sentence_batch:
        if tokenizer is None:
            indexed_tokens = sentence
        else:
            marked_text = "[CLS] " + sentence + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        if len(indexed_tokens)>512:
            indexed_tokens = indexed_tokens[-512:]
        segments_ids = [1] * len(indexed_tokens)

        tokens_tensor.append(torch.tensor(indexed_tokens))
        segments_tensor.append(torch.tensor(segments_ids))

    max_len = max(map(len, tokens_tensor))
    tokens_tensor = torch.stack([torch.cat([i, i.new_zeros(max_len - i.size(0))], 0) for i in tokens_tensor],0)
    segments_tensor = torch.stack([torch.cat([i, i.new_zeros(max_len - i.size(0))], 0) for i in segments_tensor],0)
    model.eval()
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensor, attention_mask=segments_tensor)

    return torch.split(torch.stack(encoded_layers), 1, dim=1), torch.sum(segments_tensor, axis=1), segments_tensor

def get_embeddings(encoded_layers, method=0, length=None):
    if type(encoded_layers)==list:
        encoded_layers = torch.stack(encoded_layers, dim=0)
        encoded_layers = torch.squeeze(encoded_layers, dim=1)
    else:
        new_shape = (encoded_layers.shape[0]*encoded_layers.shape[1], *encoded_layers.shape[2:])
        encoded_layers = encoded_layers.reshape(new_shape)
    token_embeddings = encoded_layers.permute(1,0,2)

    length = length or len(token_embeddings)

    token_vecs_cat = []

    if method==0:
        res = torch.zeros(token_embeddings.shape[-1]*4)

        for token in token_embeddings[:length]:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs_cat.append(cat_vec.tolist())
            res += cat_vec
    elif method==1:
        res = torch.zeros(token_embeddings.shape[-1])

        for token in token_embeddings[:length]:
            cat_vec = token[-1] + token[-2] + token[-3] + token[-4]
            token_vecs_cat.append(cat_vec.tolist())
            res += cat_vec

    return token_vecs_cat, res/length

def get_embeddings_batch(encoded_layers, lengths, method=0):
    return list(map(lambda x: get_embeddings(x[0], method, x[1]), zip(encoded_layers, lengths)))
