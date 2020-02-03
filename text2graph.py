import re

from data import DOC_1
from utils import preprocess


class Text2Graph:
    def __init__(self, text):
        self.text = text
        self.edges = {}
        self.graph = {}

    def preprocess(self, stop_filter=True, pos_filter=True):
        self.text = ". ".join(preprocess(self.text,
                                         stop_filter=stop_filter,
                                         pos_filter=pos_filter))

    @staticmethod
    def update_graph(graph, text, window):
        text += " PADPAD" * (window - 2)
        text = text.split()

        def update_collocation_weights(graph, text, i, j):
            text_ij = (text[i], text[j])

            if text_ij in graph.keys() and "PADPAD" not in text_ij:
                graph[text_ij] += 1
            elif text_ij not in graph.keys() and "PADPAD" not in text_ij:
                graph[text_ij] = 1

            return graph

        for i in range(0, len(text) - window + 1):
            for j in range(i + 1, i + window):
                graph = update_collocation_weights(graph, text, i, j)
                graph = update_collocation_weights(graph, text, j, i)

        return graph

    def transform(self, window=2):
        for sentence in re.split("[?.]", self.text):
            self.graph = self.update_graph(self.graph, sentence, window)

    def degree_centrality(self):
        node_score = {}

        for (node_1, node_2), weight_12 in self.graph.items():
            if node_1 not in node_score.keys():
                node_score[node_1] = weight_12
            else:
                node_score[node_1] += weight_12

        return sorted([(n, s) for (n, s) in node_score.items()],
                      key=lambda x: x[1],
                      reverse=True)

    def word_count(self):
        word_count = {}
        for sentence in re.split("[?.]", self.text):
            for word in sentence.split():

                if word in word_count.keys():
                    word_count[word] += 1
                else:
                    word_count[word] = 1

        return sorted([(w, s) for (w, s) in word_count.items()],
                      key=lambda x: x[1],
                      reverse=True)

    def normalized_degree_centrality(self):
        node_score = self.degree_centrality()
        node_num = len(node_score) - 1
        return [(n, s / node_num) for (n, s) in node_score]


if __name__ == "__main__":
    document = " ".join(DOC_1)

    doc = Text2Graph(document)
    doc.preprocess(stop_filter=False, pos_filter=False)
    doc.transform(window=2)
    a = doc.degree_centrality()
    b = doc.normalized_degree_centrality()

    for (node, score) in a:
        print(node, score)

    for (node, score) in b:
        print(node, score)
