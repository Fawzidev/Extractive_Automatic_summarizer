import string
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score

class Preprocessing:
    extra_stopwords = ["''", "``", "'s"]
    def __init__(self):
        return
    def sent_tokenize(self, text):
        sents = sent_tokenize(text)
        sents_filtered = []
        for s in sents:
            if s[-1] != ':' and len(s) > 10:
                sents_filtered.append(s)
        return sents_filtered

    def preprocess_text(self, text):
        sentences = self.sent_tokenize(text)
        sentences_cleaned = []
        for sent in sentences:
            words = word_tokenize(sent)
            words = [w for w in words if w not in string.punctuation]
            words = [w for w in words if w not in self.extra_stopwords]
            words = [w.lower() for w in words]

            stops = set(stopwords.words('english'))
            words = [w for w in words if w not in stops]
            sentences_cleaned.append(" ".join(words))
        return sentences_cleaned