import numpy as np
import pandas as pd
import torch
from sympy import sec
from torch import nn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

import torch.nn.functional as F
from tqdm import tqdm

# Pooling function (mean pooling)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

# Encode text using SciBERT
def encode_texts(texts, batch_size=16, max_length=512):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Take CLS token
            embeddings = model_output.last_hidden_state[:, 0, :]  # CLS pooling
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

# Main retrieval function
def retrieve_top_k(tweets_df, papers_df, k=5):
    tweet_texts = tweets_df['tweet_text'].tolist()
    paper_texts = (papers_df['title'] + " " + papers_df['abstract']).tolist()

    tweet_embeddings = encode_texts(tweet_texts)
    paper_embeddings = encode_texts(paper_texts)

    # Normalize for cosine similarity via dot product
    tweet_embeddings = F.normalize(tweet_embeddings, p=2, dim=1)
    paper_embeddings = F.normalize(paper_embeddings, p=2, dim=1)

    # Cosine similarity matrix
    sims = tweet_embeddings @ paper_embeddings.T  # Shape: (num_tweets, num_papers)
    topk = torch.topk(sims, k=k, dim=1).indices

    # Gather results
    results = []
    for i, row in tweets_df.iterrows():
        top_paper_ids = papers_df.iloc[topk[i]]['cord_uid'].tolist()
        results.append(top_paper_ids)
    tweets_df["niga"] = results
    return pd.DataFrame(results)

PATH_COLLECTION_DATA = '../subtask4b_collection_data.pkl' #MODIFY PATH
PATH_QUERY_TRAIN_DATA = '../subtask4b_query_tweets_train.tsv' #MODIFY PATH
PATH_QUERY_DEV_DATA = '../subtask4b_query_tweets_dev.tsv' #MODIFY PATH

df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
# df_collection["total"] = df_collection["title"] #df_collection["source_x"]+" "+df_collection["title"]+" "+df_collection["abstract"]

df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep = '\t')
# df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep = '\t')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter').to(device)

model.eval()

result_df = retrieve_top_k(df_query_train, df_collection, k=5)



# Evaluate retrieved candidates using MRR@k
def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (
            1 / ([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in
                                                                                         x[col_pred][:k]] else 0),
                                     axis=1)
        # performances.append(data["in_topx"].mean())
        print(data["in_topx"])
        d_performance[k] = data["in_topx"].mean()
    return d_performance


# print(df_query_train)
results_train = get_performance_mrr(df_query_train, 'cord_uid', 'niga')
print(f"Results on the train set: {results_train}")