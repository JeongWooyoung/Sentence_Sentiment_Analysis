# coding: utf-8
import numpy as np
from datetime import datetime

import arguments, os
import file_handler as fh
import evaluation_handler as eh

import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk

from gensim.test.utils import common_texts, get_tmpfile

if __name__ == '__main__':
    # Load Data
    file_name = 'wikiPlotText_m.txt'
    # {'games':games, 'corpus':corpus, 'titles':titles}
    games_data = fh.getGamesData(fh.getStoragePath()+file_name, tokenizer=nltk.word_tokenize)
    tokenized_contents = games_data['corpus_words']

    # Make Train Set

    # Vectorizing
    # sg=1(skip-gram), 0(CBOW)
    model_path = 'models/word2vec.model'
    if os.path.isfile(model_path): word2vec_model = Word2Vec.load(model_path)
    else:
        path = get_tmpfile(model_path)
        word2vec_model = Word2Vec(tokenized_contents, size=100, window=10, min_count=50, workers=8, iter=1000, sg=1)
        word2vec_model.save(model_path)
    word_vectors = word2vec_model.wv

    print(word_vectors)

    # Train CNN and Evaluation
