import re

import nltk
from nltk.stem import PorterStemmer

from graph import Graph


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

    :param text: document text that needs to be preprocessed
    :type text: str
    :param stop_filter: stopword filter status
    :type stop_filter: bool
    :param pos_filter: part of speech filter status
    :type pos_filter: bool
    :return: stemmed and preprocessed text
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


if __name__ == "__main__":
   graph = {'1': ['2', '3'],
            '2': ['1', '3'],
            '3': ['1', '2'],
            '4': ['5'],
            '5': ['4']}
   g = Graph(graph=graph)
   g.vectorize()
   print(g.node_geodesic('1', '2'))

