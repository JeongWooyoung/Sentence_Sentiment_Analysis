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

    vectorizer = Vectorizer()
    # Tokenizing
    tokenizer = vectorizer.getTokenizer(corpus['sentences'])
    tokenized_contents = vectorizer.tokenizing(corpus['sentences'], tokenizer)

    # Vectorizing
    # sg=1(skip-gram), 0(CBOW)
    model_path = 'models/word2vec_ko.model'
    word_vectors = vectorizer.vectorizing(model_path, tokenized_contents)

    print(word_vectors)
    # Make Train Set

    # Train CNN and Evaluation
