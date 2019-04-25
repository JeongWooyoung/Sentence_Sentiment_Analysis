# coding: utf-8

import os
from gensim.models import Word2Vec

from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

from gensim.test.utils import get_tmpfile

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
    def getTokenizer(self, contents):
        corpus = SentiCorpus(contents, iter_sent=True)
        word_extractor = WordExtractor(corpus)
        word_extractor.train(corpus)
        words_scores = word_extractor.extract()
        scores = {w: s.cohesion_forword for w, s in words_scores.items()}
        return LTokenizer(scores=scores)
    def tokenizing(self, contents, tokenizer):
        tokenized_contents = []
        for c in contents:
            tokenized_contents.append(tokenizer(c))
        return tokenized_contents

    def vectorizing(self, model_path, tokenized_contents):
        if os.path.isfile(model_path): word2vec_model = Word2Vec.load(model_path)
        else:
            path = get_tmpfile(model_path)
            word2vec_model = Word2Vec(tokenized_contents, size=100, window=10, min_count=50, workers=8, iter=1000, sg=1)
            word2vec_model.save(model_path)
        return word2vec_model.wv
