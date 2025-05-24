# === Required Imports ===
import math
from collections import Counter
from typing import List

import numpy as np


# === BM25 Class Definition ===
class BM25Custom:
    """
    Traditional BM25 ranking model implementation.
    """

    def __init__(self, corpus: List[List[str]], k1=1.5, b=0.75):
        """
        Initialize the BM25 model with the tokenized corpus.
        :param corpus: List of tokenized documents (each document is a list of terms).
        :param k1: Term frequency saturation parameter.
        :param b: Document length normalization parameter.
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)  # Number of documents in the corpus

        # Compute document lengths
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.N

        # Initialize term frequency (TF) and document frequency (DF)
        self.tf = []         # List of Counter dicts for each document
        self.df = {}         # Document frequency for each term

        # Build TF and DF
        for doc in corpus:
            freq = Counter(doc)
            self.tf.append(freq)
            for term in freq:
                self.df[term] = self.df.get(term, 0) + 1

        # Compute IDF for each term using smoothed formula
        self.idf = {
            term: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def score(self, query: List[str], doc_index: int) -> float:
        """
        Compute BM25 score for a single query-document pair.
        :param query: List of query terms (tokenized).
        :param doc_index: Index of the document in the corpus.
        :return: BM25 relevance score.
        """
        score = 0.0
        doc_tf = self.tf[doc_index]
        doc_len = self.doc_lens[doc_index]

        for term in query:
            if term in doc_tf:
                f = doc_tf[term]  # Term frequency in document
                idf = self.idf.get(term, 0)  # IDF can be 0 if term not seen
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator

        return score

    def rank(self, query: List[str]) -> List[float]:
        """
        Compute BM25 scores for a query against all documents in the corpus.
        :param query: List of query terms (tokenized).
        :return: List of BM25 scores corresponding to each document.
        """
        return [self.score(query, i) for i in range(self.N)]

    def get_top_n(self, query: List[str], n=5) -> List[int]:
        """
        Retrieve top-N document indices for a given query.
        :param query: List of query terms.
        :param n: Number of top results to return.
        :return: List of document indices ranked by relevance.
        """
        scores = self.rank(query)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return top_n

    def get_scores(self, query: List[str]) -> np.ndarray:
        """
        Compute BM25 scores for all documents given a tokenized query,
        using a vectorized approach consistent with ATIRE-style BM25.
        :param query: List of query terms.
        :return: NumPy array of BM25 scores per document.
        """
        scores = np.zeros(self.N)
        doc_lens = np.array(self.doc_lens)

        for term in query:
            idf = self.idf.get(term, 0)
            tf_array = np.array([doc_tf.get(term, 0) for doc_tf in self.tf])

            numerator = tf_array * (self.k1 + 1)
            denominator = tf_array + self.k1 * (1 - self.b + self.b * doc_lens / self.avgdl)

            scores += idf * (numerator / denominator)

        return scores