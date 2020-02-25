import re

import nltk
from nltk.stem import PorterStemmer


STOPWORDS = {"i", "me", "us", "you", "them", "he", "she", "him", "her",
             "their", "theirs", "it", "that",

             "can", "cant", "could", "coudnt", "shall", "should",
             "shoudnt", "shant", "will", "wont", "would", "wouldnt",

             "is", "was", "were", "werent", "a", "are", "arent"

             "maybe", "perhaps", "yes", "no", "sure", "ok", "yess",
             "absolutely", "not", "ever", "never", "often", "then", "now",
             "must",

             "at", "of", "and", "or", "for", "the", "too", "to", "its",
             "and", "as", "in", "such", "an", "into", "other",
             "used", "from", "your", "be", "if",

             "when", "how", "what", "who",

             "i.e.", "o.k.",

             "&quot", "&gt", "&lt", }


def preprocess(text, stop_filter=True, pos_filter=True):
    """Text preprocessor

    Clean text by removing punctuations, stopwords, non-noun and
    adjectives in addition to stemming.

    :return: clean text
    :rtype: str
    """

    text = text.lower()
    text = re.sub(r"[,()\n\[\]<>;:\'\{\}]", "", text)
    sentence_list = re.split("[?.]", text)

    stemmer = PorterStemmer()
    cleaned_sentence_list = []
    for sentence in sentence_list:
        clean_text = ""
        text_pos = nltk.pos_tag(sentence.split())
        for (word, tag) in text_pos:
            if pos_filter:
                if ("NN" in tag) or ("ADJ" in tag) or ("JJ" in tag):
                    clean_text += word + " "
            else:
                clean_text += word

        if stop_filter:
            sentence = " ".join([word for word in clean_text.split() if
                                 word not in STOPWORDS])

        sentence = " ".join([stemmer.stem(word) for word in
                             sentence.split()])

        cleaned_sentence_list.append(sentence)

    return cleaned_sentence_list


def node_geodesic(graph, node_1, node_2):
    """Geodesic between two nodes

    This method finds the shortest path between two nodes in a graph. This
    function can be applied to any connected or disconnected undirected graphs.

    :param graph: node relation of the undirected graph
    :type graph: dict of the form {int: list}
    :param node_1: first node of interest
    :type node_1: int
    :param node_2: second node of interest different than first node
    :type node_2: int
    :return: None (if the points are disconnected) or the shortest distance
    between the two.
    :rtype: None or int
    """

    paths = [[node_1]]
    finished_traverses = []

    while len(paths):
        new_gen_paths = []
        for path in paths:
            for adj_node in graph[path[-1]]:
                if adj_node == node_2:
                    finished_traverses.append(path + [adj_node])
                elif adj_node not in path:
                    new_gen_paths.append(path + [adj_node])
        paths = new_gen_paths

    if not len(finished_traverses):
        return None

    return min([len(path) for path in finished_traverses])


if __name__ == "__main__":
    G = [[0, 1, 1, 0, 0],
         [1, 0, 1, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0]]

    print(node_geodesic(G, 1, 2))
