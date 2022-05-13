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
from collections import defaultdict

def generate_keywords(tokenization_info, doc_to_class, num_clusters):
    docs = [x[0] for x in tokenization_info]

#     num_clusters = len(classes)

    # Sanity-check
#     print(f"Number of classes = {len(classes)}")
    print(min(doc_to_class))
    print(len(docs))
    # Prints first 10 words of first doc
    print(docs[0][0:10])
    # Prints length of first doc
    print(len(docs[0]))
    # Prints first entirety of first 10 docs
    for i in range(10):
        print(f"document #{i} = {docs[i]}\n\n\n")

    # The list cluster_sizes will hold number of documents per cluster.
    cluster_sizes = [ 0 for i in range(num_clusters) ]
    for prediction in doc_to_class:
        cluster_sizes[prediction] += 1       
    print(f"Cluster sizes = \n{cluster_sizes}\n")
    if sum(cluster_sizes) != len(docs):
        print("Inconsistent cluster sizes with total number of documents.")

    # Document-level and cluster-level dictionaries.
    # Each doc_dict stores frequency of words in that document.
    # Each cluster_dict corresponds to one cluster,
    # with one key per unique word in the documents of that cluster.
    # The value per key is how many documents from that cluster that the key appears in.
    doc_dicts = [ defaultdict(int) for i in range(len(docs)) ]
    cluster_dicts = [ defaultdict(int) for i in range(num_clusters) ]

    #Filling document-level dictionaries
    for i,doc in enumerate(tqdm(docs)):
        # Count up frequencies of unique words in ith doc
        for word in doc:
            doc_dicts[i][word] += 1
        # Sanity-check frequencies
        if sum(doc_dicts[i].values()) != len(doc):
            print("doc_dict has different number of words than actual document.")

    # Filling in cluster_dicts. cluster_dict[i] will be a dictionary mapping 
    # a unique word appearing in that cluster's documents with how many documents
    # in that cluster said word appears. Essentially document frequency of the word.
    for i,doc_dict in enumerate(tqdm(doc_dicts)):
        # Get keys as a list, basically a set of all unique words in that document, ignoring frequency.
        doc_dict_keys = list(doc_dict.keys())
        # Get assigned class/cluster
        assigned_class = doc_to_class[i]
        # Increment document frequency of that word for cluster_dicts[assigned_class].
        for key in doc_dict_keys:
            cluster_dicts[assigned_class][key] += 1
    # # Sanity-check of 
    # for cluster_dict in cluster_dicts:
    #     print(len(cluster_dict))

    #Removing words with less than args.threshold frequency from cluster-level dictionaries
    for i,cluster_dict in enumerate(tqdm(cluster_dicts)):
        keys_to_remove = []
        # If key appears in less than args.threshold of documents for this cluster, prepare for removal.
        for key in cluster_dict:
            if cluster_dict[key]/cluster_sizes[i] <= args.threshold:
                keys_to_remove.append(key)
        # Remove keys
        for key in keys_to_remove:
            cluster_dict.pop(key)

    # Now each cluster_dict will contain only keys that appear in lots of documents in that cluster.

    #Populating duplicates dictionary. Stores how many clusters each key appears in.
    duplicates_dict = defaultdict(int)
    for i,cluster_dict in enumerate(tqdm(cluster_dicts)):
        for key in cluster_dict:
            duplicates_dict[key] += 1

    #Removing duplicates from cluster-level dictionaries.
    # So each cluster-level dictionary should have keys unique to that cluster.
    for i,key in enumerate(tqdm(duplicates_dict)):
        # For keys appearing in multiple clusters...
        if duplicates_dict[key] > 1:
            # ...remove from all cluster_dict's.
            for cluster_dict in cluster_dicts:
                cluster_dict.pop(key, None)

    #Printing out each cluster-dictionary's highest-frequency keys.
    keyword_lists = [ [] for cluster in cluster_dicts ]
    for i,cluster_dict in enumerate(cluster_dicts):
        # List of (key, value) tuples for cluster_dict:
        key_to_docFreqs = list(cluster_dict.items())
        # Sort so most frequent keywords come first:
        key_to_docFreqs.sort(key=itemgetter(1), reverse=True)
        # Print info
        print(f"Class/cluster #{i} has {len(key_to_docFreqs)} generated keywords.")
        print(f"Top keywords are: {key_to_docFreqs[0:10]}")
        for j in range(10):
            keyword_lists[i].append(key_to_docFreqs[j][0])
    # Sanity-check
    print(f"keyword_lists.shape = {keyword_lists.shape}")
    return keyword_lists    

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
    
