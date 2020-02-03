import math
import pandas as pd

from text2graph import Text2Graph

from data import DOC_2
from utils import preprocess


class Corpora2Graph:
    def __init__(self, corpora):
        self.corpora = corpora
        self.vocab = set({})
        self.corpora_graphs = []

        self.word_to_index = None
        self.index_to_word = None

    def transform(self):
        for document in self.corpora:
            doc = Text2Graph(document)
            doc.preprocess(stop_filter=False, pos_filter=False)
            doc.transform(window=6)
            self.corpora_graphs.append(doc)

    def word_index(self):
        docs = preprocess(". ".join(self.corpora),
                          stop_filter=False,
                          pos_filter=False)
        vocab = set({})
        for doc in docs:
            vocab = vocab.union(set(doc.split()))

        self.vocab = sorted(vocab)

        self.word_to_index = {w: i for (i, w) in enumerate(self.vocab)}
        self.index_to_word = {i: w for (i, w) in enumerate(self.vocab)}

    def centrality_matrix(self):
        m = len(self.word_to_index.keys())
        n = len(self.corpora)

        corpora_centrality_matrix = \
            [[0 for i in range(0, n)] for j in range(0, m)]

        for j, doc in enumerate(self.corpora_graphs):
            for (word, score) in doc.degree_centrality():
                i = self.word_to_index[word]
                corpora_centrality_matrix[i][j] = score

        return corpora_centrality_matrix

    def count_matrix(self):
        m = len(self.word_to_index.keys())
        n = len(self.corpora)

        count_matrix = \
            [[0 for i in range(0, n)] for j in range(0, m)]

        for j, doc in enumerate(self.corpora_graphs):
            for (word, score) in doc.word_count():
                i = self.word_to_index[word]
                count_matrix[i][j] = score

        return count_matrix

    def tfidf(self, matrix):
        nw = len(matrix)
        nd = len(matrix[0])
        df = {i: 0 for i in range(0, nw)}
        idf = {i: 0 for i in range(0, nw)}

        for i, docs in enumerate(matrix):
            for word_count in docs:
                if word_count:
                    df[i] += 1

        for (word_index, doc_freq) in df.items():
            idf[word_index] = 1 + math.log((1.0 + nd) / (1.0 + df[word_index]))

        tfidf = [[0 for j in range(0, nd)] for i in range(0, nw)]
        for i, word_docs in enumerate(matrix):
            for j, tf in enumerate(word_docs):
                tfidf[i][j] = tf * idf[i]

        for j in range(0, nd):
            normalization_constant = \
                math.sqrt(sum([tfidf[i][j] * tfidf[i][j] for i in
                               range(0, nw)]))

            for i in range(0, nw):
                tfidf[i][j] /= normalization_constant

        return pd.DataFrame(tfidf, index=self.vocab)


if __name__ == "__main__":
    doc_list = DOC_2

    doc_graph = Corpora2Graph(corpora=doc_list)
    doc_graph.transform()
    doc_graph.word_index()
    df_1 = doc_graph.tfidf(doc_graph.centrality_matrix())
    df_2 = doc_graph.tfidf(doc_graph.count_matrix())
    print(df_1)
    print(df_2)


