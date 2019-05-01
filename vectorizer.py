# coding: utf-8

import os
from gensim.models import Word2Vec

from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

import file_handler as fh

class SentiCorpus:
    def __init__(self, contents, iter_sent = False):
        self.iter_sent = iter_sent
        self.lines = contents
    def __iter__(self):
        for line in self.lines:
            if line == '\n': continue
            yield line
    def __len__(self):
        return len(self.lines)
class Vectorizer:
    def __init__(self):
        self.tokenizer = None
        self.trained = False
    def getTokenizer(self, contents):
        corpus = SentiCorpus(contents, iter_sent=True)
        word_extractor = WordExtractor(corpus)
        word_extractor.train(corpus)
        words_scores = word_extractor.extract()
        scores = {w: s.cohesion_forward for w, s in words_scores.items()}
        return LTokenizer(scores=scores)
    def tokenizing(self, contents):
        if self.tokenizer is None: self.tokenizer = self.getTokenizer(contents)

        tokenized_contents = []
        for c in contents:
            tokenized_contents.append(self.tokenizer(c))
        return tokenized_contents
    def vectorize(self, model_path, contents, training=True):
        if training:
            tokenized_contents = self.tokenizing(contents)

            word2vec_model = Word2Vec(tokenized_contents, size=20, window=6, min_count=2, workers=8, iter=1000, sg=1)
            dir_path = model_path[:model_path.rfind('/')]

            if not os.path.isdir(dir_path): fh.makeDirectories(dir_path)
            word2vec_model.save(model_path)
            self.trained = True
        elif os.path.isfile(model_path):
            tokenized_contents = self.tokenizing(contents)
            word2vec_model = Word2Vec.load(model_path)
        else: return None

        # return word2vec_model.wv
        word_vectors = word2vec_model.wv
        senti_dict = fh.getSentiDictionary()
        # 추후 작업필요(기본형 기준 매칭 정확도 개선필요)
        word_vectors = {w:word_vectors[w].tolist()+[senti_dict[w] if w in senti_dict.keys() else .0] for w in word_vectors.vocab.keys()}

        vectorized_contents = []
        for s in tokenized_contents :
            vectorized_content = []
            for w in s:
                if w in word_vectors.keys():
                    vectorized_content.append(word_vectors[w])
            if len(vectorized_content) > 0: vectorized_contents.append(vectorized_content)
        return vectorized_contents
