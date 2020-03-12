import math
import pandas as pd

from text2graph import Text2Graph

from data import DOC_2
from utils import preprocess


class Corpora2Graph:
    def __init__(self, corpora):
        """Initialize set of documents (corpora)

        :param corpora: list of documents
        :type corpora: list of str
        """

        self.corpora = corpora
        self.vocab = set({})
        self.corpora_graphs = []

        self.word_to_index = None
        self.index_to_word = None

    def transform(self):
        """Transform corpora

        Transform each document into a graph using Text2Graph
        """

        for document in self.corpora:
            doc = Text2Graph(document)
            doc.preprocess(stop_filter=False, pos_filter=False)
            doc.transform(window=6)
            self.corpora_graphs.append(doc)

    def word_index(self):
        """Index corpora vocabulary

        After preprocessing, map corpora vocabulary to indices corresponding to
        the matrix representation.
        """

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
        """Matrix representation of word centrality

        Create a matrix representation with the degree centrality of the
        corpora.

        :return: corpora centrality matrix
        :rtype: list of list
        """

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
        """Matrix representation of word frequency

        Create a matrix representation using word frequency in the corpora.

        :return: corpora word frequency matrix
        :rtype: list of list
        """

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
        """Tfidf matrix

        Create term frequency-inverse document frequency matrix
        representation of the corpora matrix representation which could be
        based on frequency based OR centrality based measures.

        :param matrix: some metric representation of the corpora
        :type matrix: list of list
        :return: comparison of tfidf scores applied tovarious word count
        measures.
        :rtype: pd.DataFrame
        """

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


