#%%
from pytorch_pretrained_bert import BertModel
from bert_tokenizer import BertTokenizer
import torch

# %%
MODEL = "./biobert_large"
bio_tokenizer = BertTokenizer.from_pretrained("models/vocab.txt")
bio_model = BertModel.from_pretrained(MODEL)

# %%
sentence = "Herbal tea hepatotoxicity"
marked_text = "[CLS] " + sentence + " [SEP]"
tokenized_text = bio_tokenizer.tokenize(marked_text)
indexed_tokens = bio_tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.Tensor([indexed_tokens]).long()
segments_tensor = torch.Tensor([[1] * len(indexed_tokens)]).long()
mask = segments_tensor.clone().detach()

# %%
bio_model.eval()
traced = torch.jit.trace(bio_model, (tokens_tensor, segments_tensor, mask))
# print(bio_model.forward(tokens_tensor, segments_tensor, mask))


# %%
traced.save("biobert_large/biobert_batch.pt")