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


def main(args):

    cluster_data_dir = f"../data/intermediate_data/{args.dataset_name}/data.pca{args.pca}.clus{args.cluster}.bbu-12.mixture-100.42_{args.mask}_{args.confidence}.pk"
    with open(cluster_data_dir, "rb") as f:
        cluster_data = pk.load(f)
        doc_to_class = cluster_data["documents_to_class"]
        distances = cluster_data["distance"]

        # docs_dir = f"../data/datasets/{args.dataset_name}/dataset.txt"
        # docs_data_dir = f"../data/intermediate_data/{args.dataset_name}/dataset.pk"
        # with open(docs_data_dir, "rb") as f:
        #     docs_data = pk.load(f)
        #     docs = docs_data["cleaned_text"]
        # docs = [x.lower() for x in docs]
        # docs = np.loadtxt(docs_dir, dtype='str')
        # docs = list(docs)
    
        #[Weiyu's changes]
        # for i in range(5):
        #     print(doc_to_class[i])
        # print(doc_to_class.shape, type(doc_to_class)) #added by weiyu

    docs_data_dir = f"../data/intermediate_data/{args.dataset_name}/tokenization_lm-bbu-12.pk"  
    with open(docs_data_dir, "rb") as f:
        docs_data = pk.load(f)
        tokenization_info = docs_data["tokenization_info"]
    docs = [x[0] for x in tokenization_info]


    classes_dir = f"../data/datasets/{args.dataset_name}/classes.txt"
    classes = np.loadtxt(classes_dir, dtype='str')
    classes = list(classes)

    num_clusters = len(classes)
    class_integers = [x for x in range(len(classes))]
        # print(class_integers)
        # print(doc_to_class.shape)
        # print(len(docs))
    print(f"Number of classes = {len(classes)}")
        # print(min(doc_to_class))
        # print(len(docs))
        # print(docs[0][0:10])
        # print(len(docs[0]))
        # for i in range(10):
        #     print(f"document #{i}")
        #     print(docs[i])
        #     print()
        #     print()

    doc_dicts = [ {} for i in range(len(docs)) ]
    cluster_dicts = [ {} for i in range(num_clusters) ]
    cluster_sizes = [ 0 for i in range(num_clusters) ]
    print("Cluster sizes:")
    for prediction in doc_to_class:
        cluster_sizes[prediction] += 1
    print(cluster_sizes)
        # print(sum(cluster_sizes))

    #Filling document-level dictionaries
    for i,doc in enumerate(tqdm(docs)): #looping through all documents
        for word in doc:
            if word not in doc_dicts[i]:
                doc_dicts[i][word] = 1
            else:
                doc_dicts[i][word] += 1        
                # for word in doc:
                #     if word not in doc_dicts[i]:
                #         doc_dicts[i][word] = 1
                #     else:
                #         doc_dicts[i][word] += 1

    #Filling cluster-level dictionaries
    for i,doc_dict in enumerate(tqdm(doc_dicts)):
        doc_dict_keys = list(doc_dict)
        doc_predicted_class = doc_to_class[i]
        for key in doc_dict_keys:
            if key not in cluster_dicts[doc_predicted_class]:
                cluster_dicts[doc_predicted_class][key] = 1
            else:
                cluster_dicts[doc_predicted_class][key] += 1
    for c_dict in cluster_dicts:
        print(len(c_dict))

    #Removing words with <5% frequency from cluster-level dictionaries
    for i,cluster_dict in enumerate(tqdm(cluster_dicts)):
        keys_to_remove = []
        for key in cluster_dict:
            if cluster_dict[key]/cluster_sizes[i] <= args.threshold:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            cluster_dict.pop(key)

    #Populating duplicates dictionary.
    duplicates_dict = {}
    for i,cluster_dict in enumerate(tqdm(cluster_dicts)):
        for key in cluster_dict:
            if key not in duplicates_dict:
                duplicates_dict[key] = 1
            else:
                duplicates_dict[key] += 1

    #Removing duplicates from cluster-level dictionaries.
    for i,key in enumerate(tqdm(duplicates_dict)):
        if duplicates_dict[key] > 1:
            for cluster_dict in cluster_dicts:
                cluster_dict.pop(key, None)

    #Printing out each cluster-dictionary's highest-frequency keys.
    for i,cluster_dict in enumerate(cluster_dicts):
        key_val_tuples = list(cluster_dict.items())
        print(classes[i])
        print(f"key_val_tuples length = {len(key_val_tuples)}")
        key_val_tuples.sort(key=itemgetter(1), reverse=True)
        
        print(key_val_tuples[0:10])
        print()
        print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--mask", type=str)
    parser.add_argument("--cluster", type=str)
    parser.add_argument("--pca", type=str, default=64)
    parser.add_argument("--confidence", type=str, default="0.15")
    args = parser.parse_args()
    print(args)
    print(vars(args))
    main(args)
