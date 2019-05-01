# coding: utf-8
import numpy as np
from datetime import datetime

import arguments, os
import file_handler as fh
import evaluation_handler as eh
from vectorizer import Vectorizer

import tensorflow as tf

if __name__ == '__main__':
    # Load Data
    senti_dir = './KOSAC_sample/'
    # {'sentences':sentences, 'tokens':tokens, 'polarity':polarity}
    corpus = fh.getSentiCorpus(senti_dir)
    contents = corpus['sentences']

    vec = Vectorizer()
    tokenized_contents = vec.tokenizing(contents)
    # Vectorizing
    # sg=1(skip-gram), 0(CBOW)
    model_path = 'models/word2vec_ko.model'
    vectorized_contents = vec.vectorize(model_path, contents)

    print(vectorized_contents)
    # Make Train Set
    # Embedding by convolution

    # Train ML and Evaluation
