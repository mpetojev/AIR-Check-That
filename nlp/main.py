import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import torch.nn.functional as F

# Pooling function (CLS pooling)
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0, :]

# Encode text using SPECTER

def encode_texts(texts, batch_size=128, max_length=512):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            embeddings = cls_pooling(model_output)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

# Retrieval function (dense retrieval)
def retrieve_top_k(tweets_df, papers_df, k=100):
    tweet_texts = tweets_df['tweet_text'].tolist()
    paper_texts = (papers_df['title'] + " " + papers_df['abstract']).tolist()

    tweet_embeddings = encode_texts(tweet_texts)
    paper_embeddings = encode_texts(paper_texts)

    tweet_embeddings = F.normalize(tweet_embeddings, p=2, dim=1)
    paper_embeddings = F.normalize(paper_embeddings, p=2, dim=1)

    sims = tweet_embeddings @ paper_embeddings.T  # Cosine similarities
    topk = torch.topk(sims, k=k, dim=1).indices

    return topk

# Cross-encoder re-ranking
def rerank_with_cross_encoder(tweets_df, papers_df, topk_indices, batch_size=512):
    reranked_results = []
    all_pairs = []
    pair_mapping = []

    for i, topk_paper_indices in enumerate(topk_indices):
        query = tweets_df.iloc[i]['tweet_text']
        candidates = papers_df.iloc[topk_paper_indices]

        pairs = [(query, title + " " + abstract) for title, abstract in zip(candidates['title'], candidates['abstract'])]

        all_pairs.extend(pairs)
        pair_mapping.extend([(i, j) for j in range(len(pairs))])

    print(f"Total pairs to rerank: {len(all_pairs)}")

    all_scores = []
    for i in tqdm(range(0, len(all_pairs), batch_size), desc="Cross-Encoder reranking"):
        batch = all_pairs[i:i+batch_size]
        batch_scores = cross_encoder.predict(batch)
        all_scores.extend(batch_scores)

    from collections import defaultdict
    temp = defaultdict(list)

    for (tweet_idx, cand_idx), score in zip(pair_mapping, all_scores):
        temp[tweet_idx].append((cand_idx, score))

    for i in range(len(tweets_df)):
        candidate_scores = temp[i]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        candidate_indices = [c[0] for c in candidate_scores]
        top_paper_ids = papers_df.iloc[topk_indices[i][candidate_indices]]['cord_uid'].tolist()
        reranked_results.append(top_paper_ids)

    return reranked_results

# Evaluation MRR@k
def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (
            1 / ([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in
                                                                                         x[col_pred][:k]] else 0),
                                     axis=1)
        d_performance[k] = data["in_topx"].mean()
    return d_performance

# ========== PATHS ===========
PATH_COLLECTION_DATA = '../subtask4b_collection_data.pkl'  # MODIFY PATH
PATH_QUERY_TRAIN_DATA = '../subtask4b_query_tweets_test.tsv'  # MODIFY PATH
PATH_QUERY_DEV_DATA = '../subtask4b_query_tweets_dev.tsv'  # MODIFY PATH

# ========== LOAD MODELS ===========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading SPECTER model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter").to(device)
model.eval()

print("Loading Cross-Encoder...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ========== LOAD DATA ===========
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)  # use more if you want
# print(df_collection.columns.values)
# print(df_collection.iloc[0])
# Only use title + abstract
# df_collection['total'] = df_collection['title'] + ' ' + df_collection['abstract']

df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep='\t')

# ========== RETRIEVAL + RERANKING ===========
print("Retrieving candidates with SPECTER...")
topk_indices = retrieve_top_k(df_query_train, df_collection, k=100)

print("Reranking with Cross-Encoder...")
reranked_results = rerank_with_cross_encoder(df_query_train, df_collection, topk_indices)

df_query_train["niga"] = reranked_results

df_query_train[['post_id', 'niga']].to_csv('predictions.tsv', index=None, sep='\t')

# ========== EVALUATION ===========
print("Evaluating performance...")
results_train = get_performance_mrr(df_query_train, 'cord_uid', 'niga')
print(f"Results on the train set: {results_train}")

#
# Results on the train set: {1: np.float64(0.47568661012993074), 5: np.float64(0.5165162478280038), 10: np.float64(0.5201303444690203)}