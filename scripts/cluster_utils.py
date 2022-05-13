from document_representations import weight_sentence
from utils import (INTERMEDIATE_DATA_FOLDER_PATH, MODELS,
                   cosine_similarity_embedding, cosine_similarity_embeddings,
                   evaluate_predictions, tensor_to_numpy)

import argparse
import math
import os
import pickle as pk
import random
import re

import numpy as np
import scipy.stats
from scipy import linalg
from tqdm import tqdm
from operator import itemgetter

def generate_keywords():
    # TODO

def generate_class_representation(keywords, lm_type, layer):

    static_repr_path = os.path.join(data_folder, f"static_repr_lm-{lm_type}-{layer}.pk")
    with open(static_repr_path, "rb") as f:
        vocab = pk.load(f)
        static_word_representations = vocab["static_word_representations"]
        word_to_index = vocab["word_to_index"]

    print("Finish reading data")

    class_words_representations = [[static_word_representations[word_to_index[word]]]
                                   for word in keywords]

    cls_repr = average_with_harmonic_series(class_words_representations)

    return cls_repr

def generate_doc_representations(class_representations, attention_mechanism,lm_type, layer):
    
    static_repr_path = os.path.join(data_folder, f"static_repr_lm-{lm_type}-{layer}.pk")
    with open(static_repr_path, "rb") as f:
        vocab = pk.load(f)

    with open(os.path.join(data_folder, f"tokenization_lm-{lm_type}-{layer}.pk"), "rb") as f:
        tokenization_info = pk.load(f)["tokenization_info"]

    model_class, tokenizer_class, pretrained_weights = MODELS[args.lm_type]
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    model.eval()
    model.cuda()
    
    document_representations = []
    for i, _tokenization_info in tqdm(enumerate(tokenization_info), total=len(tokenization_info)):
        document_representation = weight_sentence(model,
                                                  vocab,
                                                  _tokenization_info,
                                                  class_representations,
                                                  attention_mechanism,
                                                  layer)
        document_representations.append(document_representation)
    document_representations = np.array(document_representations)

    return document_representations
    
