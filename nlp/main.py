import numpy as np
import pandas as pd
import torch
from sympy import sec
from torch import nn
from transformers import AutoModel, AutoTokenizer


# Evaluate retrieved candidates using MRR@k
def get_performance_mrr(data, col_gold, col_pred, list_k = [1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        #performances.append(data["in_topx"].mean())
        d_performance[k] = data["in_topx"].mean()
    return d_performance

def split_dataframe(df, chunk_size = 64):
    chunks = list()
    num_chunks = len(df)
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

cos = nn.CosineSimilarity(dim=1)

def get_top_cord_uids(query):
    # print(query)
    inputs = tokenizer(query,
                       padding=True,
                       truncation=True,
                       return_tensors='pt',
                       max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    # Expand dimensions of attention mask (so it matches token_embeddings shape)
    attention_mask = attention_mask.unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)

    # Compute mean pooling: sum(token_embeddings * mask) / sum(mask)
    tweet_emb = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    sims = cos(tweet_emb.unsqueeze(0), paper_embeddings)
    top5_indices = torch.topk(sims, k=5).indices.tolist()
    print(top5_indices)
    top5_papers = df_collection.iloc[top5_indices[0]]['cord_uid'].tolist()
    return top5_papers
#     text2bm25top = {}
#     if query in tweet.keys():
#         return text2bm25top[query]
#     else:
#         tokenized_query = query.split(' ')
#         doc_scores = bm25.get_scores(tokenized_query)
#         indices = np.argsort(-doc_scores)[:5]
#         bm25_topk = [cord_uids[x] for x in indices]
#
#         text2bm25top[query] = bm25_topk
#         return bm25_topk
# for i, tweet_emb in enumerate(tweet_embeddings):
#     sims = cos(tweet_emb.unsqueeze(0), paper_embeddings)  # Shape: (num_papers,)
#     top5_indices = torch.topk(sims, k=5).indices.tolist()
#     top5_papers = papers_df.iloc[top5_indices]['paper_id'].tolist()
#
#     results.append({
#         'tweet_id': tweets_df.iloc[i]['tweet_id'],
#         'top5_paper_ids': top5_papers
#     })

PATH_COLLECTION_DATA = '../subtask4b_collection_data.pkl' #MODIFY PATH
PATH_QUERY_TRAIN_DATA = '../subtask4b_query_tweets_train.tsv' #MODIFY PATH
PATH_QUERY_DEV_DATA = '../subtask4b_query_tweets_dev.tsv' #MODIFY PATH

df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
df_collection["total"] = df_collection["title"] #df_collection["source_x"]+" "+df_collection["title"]+" "+df_collection["abstract"]

df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep = '\t')
df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep = '\t')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)

paper_embeddings = None
iter = 0
lena = int(len(df_collection)/64)+1
for batch in split_dataframe(df_collection):
    try:
        inputs = tokenizer(batch['total'].tolist(),
                           padding=True,
                           truncation=True,
                           return_tensors='pt',
                           max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Expand dimensions of attention mask (so it matches token_embeddings shape)
        attention_mask = attention_mask.unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)

        # Compute mean pooling: sum(token_embeddings * mask) / sum(mask)
        sentence_embeddings = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        if paper_embeddings is None:
            paper_embeddings = sentence_embeddings
        else:
            paper_embeddings = torch.cat((paper_embeddings, sentence_embeddings), dim=0)
        # paper_embeddings.extend(sentence_embeddings.tolist())

        iter += 1
        if iter % 10 == 0:
            print(f"Heartbeat {iter}/{lena}")
    except Exception:
        pass

    # if iter == 10:
        # break
    # break
    # torch.cat(sentence_embeddings, dim=0)
    # print(paper_embeddings.shape)  # (batch_size, hidden_dim)
# Encode tweets
# tweet_embeddings = model.encode(df_query_train['tweet_text'].tolist(), convert_to_tensor=True)
# Encode papers
# paper_embeddings = model(df_collection['total'].tolist(), convert_to_tensor=True)


print(f"{len(paper_embeddings)}x{len(paper_embeddings[0])}")
# df_query_train.apply(lambda x: get_top_cord_uids(x))
# Extract token embeddings

df_query_train['nlpTop5'] = df_query_train['tweet_text'].apply(lambda x: get_top_cord_uids(x))

def get_performance_mrr(data, col_gold, col_pred, list_k = [1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        #performances.append(data["in_topx"].mean())
        d_performance[k] = data["in_topx"].mean()
    return d_performance
results_train = get_performance_mrr(df_query_train, 'cord_uid', 'nlpTop5')


print(f"Results on the train set: {results_train}")
