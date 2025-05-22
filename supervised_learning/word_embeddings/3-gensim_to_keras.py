#!/usr/bin/env python3
"""
    Defines a function that converts a gensim word2vec model to a Keras Embedding layer
"""

def gensim_to_keras(model):
    '''
        Converts a gensim word2vec model to a Keras Embedding layer
    '''
    return model.wv.get_keras_embedding(train_embeddings=True)
