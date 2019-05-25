# coding: utf-8
import arguments
import numpy as np

import file_handler as fh
import evaluation_handler as eh
import learn_handler as lh
from vectorizer import Vectorizer

if __name__ == '__main__':
    # Load Data
    # senti_dir = './KOSAC_sample/'
    # # {'sentences':sentences, 'tokens':tokens, 'polarity':polarity}
    # corpus = fh.getSentiCorpus(senti_dir)
    # contents = corpus['sentences']

    comments = fh.getComments()
    contents = comments['sentences']

    # Vectorizing
    # vec = Vectorizer()
    # tokenized_contents = vec.tokenizing(contents)
    # # sg=1(skip-gram), 0(CBOW)
    # model_path = 'models/word2vec_ko.model'
    # vectorized_contents = vec.vectorize(model_path, contents, dims=100)
    #
    # fh.saveVectorizedContents(vectorized_contents)
    vectorized_contents = fh.loadVectorizedContents()

    # Train ML and Evaluation
    args = arguments.parse_args('Bidirectional_LSTM')
    eh.evaluations(vectorized_contents, comments['scores'], lh.Bidirectional_LSTM(args))
