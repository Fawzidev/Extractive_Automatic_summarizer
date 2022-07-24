import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import Preprocessing
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import gensim.models.keyedvectors as word2vec


def average_score(scores):
    score = 0
    count = 0
    for s in scores:
        if s > 0:
            score += s
            count += 1
    if count > 0:
        score /= count
        return score
    else:
        return 0

def get_max_length(sentences):
    max_length = 0
    for s in sentences:
        l = len(s.split())
        if l > max_length:
            max_length = l
    return max_length

class CentroidSummarizer(Preprocessing.Preprocessing):

    def __init__(self):
        word2vec_model_path = 'wiki.en.vec'
        self.word2vec = word2vec.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True,
                                                                   unicode_errors='ignore')
        self.index2word_set = set(self.word2vec.wv.index2word)
        self.word_vectors = dict()
        self.topic_threshold = 0.3
        self.subtract_centroid = False
        self.bow_param = 0
        self.length_param = 0
        self.position_param = 0
        self.sim_threshold = 0.95
        
        count = 0
        self.centroid_space = np.zeros(self.word2vec.vector_size, dtype="float32")
        for w in self.index2word_set:
            self.centroid_space = self.centroid_space + self.word2vec[w]
            count += 1
        if count != 0:
            self.centroid_space = np.divide(self.centroid_space, count)
        return

    def get_bow(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0
        return tfidf, centroid_vector

    def get_topic_idf(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())

        feature_names = vectorizer.get_feature_names()
        word_list = []
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] > 0.3:
                word_list.append(feature_names[i])

        return word_list

    def word_vectors_cache(self, sentences):
        self.word_vectors = dict()
        for s in sentences:
            words = s.split()
            for w in words:
                if w in self.index2word_set:
                    if self.subtract_centroid:
                        self.word_vectors[w] = (self.word2vec[w] - self.centroid_space)
                    else:
                        self.word_vectors[w] = self.word2vec[w]
        return

    def compose_vectors(self, words):
        composed_vector = np.zeros(self.word2vec.vector_size, dtype="float32")
        word_vectors_keys = set(self.word_vectors.keys())
        count = 0
        for w in words:
            if w in word_vectors_keys:
                composed_vector = composed_vector + self.word_vectors[w]
                count += 1
        if count != 0:
            composed_vector = np.divide(composed_vector, count)
        return composed_vector

    def summarize(self, text, limit):
        raw_sentences = self.sent_tokenize(text)
        clean_sentences = self.preprocess_text(text)
        centroid_words = self.get_topic_idf(clean_sentences)

        print("*** CENTROID WORDS ***")
        print(len(centroid_words), centroid_words)

        self.word_vectors_cache(clean_sentences)
        centroid_vector = self.compose_vectors(centroid_words)

        tfidf, centroid_bow = self.get_bow(clean_sentences)
        max_length = get_max_length(clean_sentences)

        sentences_scores = []
        for i in range(len(clean_sentences)):
            scores = []
            words = clean_sentences[i].split()
            sentence_vector = self.compose_vectors(words)

            scores.append(Preprocessing.similarity(sentence_vector, centroid_vector))
            scores.append(self.bow_param * Preprocessing.similarity(tfidf[i, :], centroid_bow))
            scores.append(self.length_param * (1 - (len(words) / max_length)))
            scores.append(self.position_param * (1 / (i + 1)))

            score = average_score(scores)
            sentences_scores.append((i, raw_sentences[i], score, sentence_vector))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
        count = 0
        sentences_summary = []

        for s in sentence_scores_sort:
            if s[0] == 0:
                sentences_summary.append(s)
                count += 1
                sentence_scores_sort.remove(s)
                break

        for s in sentence_scores_sort:
            if count > limit:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = Preprocessing.similarity(s[3], ps[3])
                if sim > self.sim_threshold:
                    include_flag = False
            if include_flag:
                sentences_summary.append(s)
                count += 1

        sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)
        summary = " ".join([s[1] for s in sentences_summary])

        return summary