import argparse
import json
import math
import os
import pickle as pk
import random
import re

import numpy as np
import scipy.stats
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.preprocessing import normalize
from tqdm import tqdm

from utils import (INTERMEDIATE_DATA_FOLDER_PATH, cosine_similarity_embedding,
                   cosine_similarity_embeddings, evaluate_predictions,
                   most_common, pairwise_distances)
from cluster_utils import (generate_keywords, generate_class_representation, 
                    generate_doc_representations)


def importData(data_dir, lm_type, layer, document_repr_type):
#     with open(os.path.join(data_dir, "dataset.pk"), "rb") as f:
#         dictionary = pk.load(f)
#         class_names = dictionary["class_names"]
    with open(os.path.join(data_dir, f"tokenization_lm-{lm_type}-{layer}.pk"), "rb") as f:
        tokenization_info = pk.load(f)["tokenization_info"]
    with open(os.path.join(data_dir, f"document_repr_lm-{lm_type}-{layer}-{document_repr_type}.pk"), "rb") as f:
        dictionary = pk.load(f)
        class_representations = dictionary["class_representations"]
        document_representations = dictionary["document_representations"]
        raw_document_representations = dictionary["raw_document_representations"]
    with open(os.path.join(data_dir, f"static_repr_lm-{lm_type}-{layer}.pk"), "rb") as f:
        dictionary = pk.load(f)
        vocab_words = dictionary["vocab_words"]
    return tokenization_info, class_representations, document_representations, raw_document_representations, vocab_words
  
#Partitions document representations into low and high confidence groups
def partitionDataset(threshold, doc_reps, known_class_reps):
    #Matrix of cosine similarities with respect to known class representations
    cosine_similarities = cosine_similarity_embeddings(doc_reps, known_class_reps) 
    document_class_assignment = np.argmax(cosine_similarities, axis=1) #Cosine similarity predictions
    low_conf_docs = []
    high_conf_docs = []
    for i,doc_rep in enumerate(doc_reps):
        doc_tuple = (doc_rep, i)
        cosine_predicted_class = document_class_assignment[i] #Prediction out of known classes
        doc_max_similarity = cosine_similarities[i][cosine_predicted_class] #Gets actual similarity
        if doc_max_similarity >= threshold:
            high_conf_docs.append(doc_tuple)
        else:
            low_conf_docs.append(doc_tuple)
    #Check partition sizes
    print(f"Confidence threshold = {threshold}")
    print(f"Number of low confidence docs = {len(low_conf_docs)}")
    print(f"Number of high confidence docs = {len(high_conf_docs)}")
    return low_conf_docs, high_conf_docs, document_class_assignment

def replace_with_raw(low_conf_docs, raw_doc_reps):
    raw_low_conf_docs = []
    for doc, index in low_conf_docs:
        raw_low_conf_docs.append(raw_doc_reps[index])
    return raw_low_conf_docs

def finalGMM(final_doc_representations, final_class_representations, num_expected, random_state):
    cosine_similarities = cosine_similarity_embeddings(final_doc_representations, final_class_representations)
    document_class_assignment = np.argmax(cosine_similarities, axis=1)
    document_class_assignment_matrix = np.zeros((final_doc_representations.shape[0], num_expected))
    for i in range(final_doc_representations.shape[0]):
        document_class_assignment_matrix[i][document_class_assignment[i]] = 1.0
    document_class_assignment_summary = np.sum(document_class_assignment_matrix, axis=0)
    print(f"GMM number of initializations per class: {document_class_assignment_summary}")
    # Performing final GMM
    gmm = GaussianMixture(n_components=num_expected, covariance_type='tied',
                          random_state=random_state,
                          n_init=999, warm_start=True)
    gmm.converged_ = "HACK"
    gmm._initialize(final_doc_representations, document_class_assignment_matrix)
    gmm.lower_bound_ = -np.infty
    gmm.fit(final_doc_representations)

    documents_to_class = gmm.predict(final_doc_representations)
    centers = gmm.means_
    distance = -gmm.predict_proba(final_doc_representations) + 1
    return documents_to_class, centers, distance


def main(dataset_name,
         pca,
         cluster_method,
         rep_type,
         lm_type,
         document_repr_type,
         attention_mechanism,
         layer,
         random_state,
         num_expected):
    
    '''#####################
    INITIAL DATA PREPARATION
    #####################'''
    # Save arguments
    do_pca = pca != 0  # pca = 0 means no pca
    save_dict_data = {}
    save_dict_data["dataset_name"] = dataset_name
    save_dict_data["pca"] = pca
    save_dict_data["cluster_method"] = cluster_method
    save_dict_data["lm_type"] = lm_type
    save_dict_data["document_repr_type"] = document_repr_type
    save_dict_data["random_state"] = random_state
    # File names and directories
    naming_suffix = f"pca{pca}.clus{cluster_method}.{lm_type}-{layer}.{document_repr_type}.{random_state}"
    data_dir = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name)
    print(f"naming_suffix: {naming_suffix}")
    print(f"data_dir: {data_dir}")
    # Import needed data
    tokenization_info, class_representations, document_representations, raw_document_representations, vocab_words = importData(data_dir, lm_type, layer, document_repr_type)  
    # Perform PCA on representations    
    class_representations_no_pca = class_representations
    if do_pca:
        _pca = PCA(n_components=pca, random_state=random_state)
        document_representations        = _pca.fit_transform(document_representations)
        raw_document_representations    = _pca.transform(raw_document_representations)
        class_representations           = _pca.transform(class_representations)
        print(f"Explained variance: {sum(_pca.explained_variance_ratio_)}")

    '''#####################
     MAIN LOOP: Generations
    #####################'''      
    for gen in range(1,4):
        print(f"Starting generation #{gen}")
        # Partitioning dataset 
        print(f"Partitioning documents gen{gen}")
        low_conf_docs, high_conf_docs, document_class_assignment = partitionDataset(0.10, document_representations, class_representations)

        # Cluster low-confidence documents 
        print(f"Clustering lowconf gen{gen}")
        low_conf_doc_reps = replace_with_raw(low_conf_docs, raw_document_representations)
        gmm = GaussianMixture(n_components=num_expected, covariance_type='tied', random_state=random_state, n_init=30, warm_start=False, verbose=0)
        gmm.fit(low_conf_doc_reps) 
        low_conf_doc_predictions = gmm.predict(low_conf_doc_reps) 

        # Generate keywords
        print(f"Generating keywords gen{gen}")
        low_conf_indices = [ doc_tuple[1] for doc_tuple in low_conf_docs ] # Grab indices with respect to all documents of low_conf_docs
        cluster_keywords = generate_keywords(tokenization_info, low_conf_doc_predictions, low_conf_indices, num_expected, vocab_words)
        for keywords in cluster_keywords:
            print(f"Cluster Words :{keywords}")

        # Generating class representations
        print(f"Generating low-conf class reps gen{gen}")
        low_conf_class_reps = [ generate_class_representation(keywords, lm_type, layer, data_dir) for keywords in cluster_keywords ]
        if len(low_conf_class_reps) != num_expected:
            print("Incorrect number of generated class representations.")
            return
        low_conf_class_reps = np.array(low_conf_class_reps)
        # Selecting class representations
        print(f"Matching new class reps gen{gen}")
        from scipy.optimize import linear_sum_assignment as hungarian
        class_rep_similarity = cosine_similarity_embeddings(low_conf_class_reps, class_representations_no_pca)
        row_ind, col_ind = hungarian(class_rep_similarity, maximize=False)
        # user_chosen_tossout = [int(x) for x in input("Choose which clusters to toss out:\n").split()]
        # ^ Allows for user to choose which clusters to keep and which to toss out.
        # row_ind is list of cluster numbers to be tossed out. Remaining row indices correspond to our new class representations
        generated_class_reps = [ low_conf_class_reps[i] for i in range(num_expected) if i not in row_ind ]
        # Finalizing class representations for next generation of document representations
        final_class_representations = np.concatenate((class_representations_no_pca, generated_class_reps))
        for i in range(num_expected):
            if i not in row_ind:
                print(f"Keeping cluster #{i} with keywords: {cluster_keywords[i]}")
        print(f"final_class_representations.shape = {final_class_representations.shape}") # Should be (num_expected)x(768)
        if final_class_representations.shape != (num_expected,768):
            print("final_class_representations shape is wrong.")
            return

        # Recalculate new document representations for all documents, these are class aligned with both the known and generated classes
        print(f"Generating doc reps gen{gen}")
        final_doc_representations = generate_doc_representations(final_class_representations, attention_mechanism, lm_type, layer, data_dir)
        print(f"Saving gen{gen} representations")
        save_dict_data[f"class_representations_gen{gen}"] = final_class_representations
        save_dict_data[f"doc_representations_gen{gen}"] = final_doc_representations
        # Initialize representations for next generation
        class_representations = final_class_representations
        document_representations = final_doc_representations
        if do_pca:
            print(f"PCA on gen{gen} class/doc reps")
            _pca = PCA(n_components=pca, random_state=random_state)
            final_doc_representations = _pca.fit_transform(final_doc_representations)
            final_class_representations = _pca.transform(final_class_representations)
            print(f"Final explained variance: {sum(_pca.explained_variance_ratio_)}")

        # Final GMM clustering results
        print(f"Final GMM on gen{gen} doc reps")
        documents_to_class, centers, distance = finalGMM(final_doc_representations, final_class_representations, num_expected, random_state)
        save_dict_data[f"documents_to_class_gen{gen}"] = documents_to_class
        save_dict_data[f"centers_gen{gen}"] = centers
        save_dict_data[f"distance_gen{gen}"] = distance
        save_dict_data["num_generations"] = gen
        # Save after every generation, overwriting previous generations' pickle files is ok.
        with open(os.path.join(data_dir, f"data.{naming_suffix}_multiGen.pk"), "wb") as f:
            pk.dump(save_dict_data, f)
        print(f"Finished generation #{gen}")            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="NYT-Topics")
    parser.add_argument("--pca", type=int, default=64, help="number of dimensions projected to in PCA, "
                                                            "-1 means not doing PCA.")
    parser.add_argument("--cluster_method", choices=["gmm", "kmeans"], default="gmm")
    parser.add_argument("--rep_type", choices=["generated", "raw"], default="generated")
    # language model + layer
    parser.add_argument("--lm_type", default="bbu")
    # attention mechanism + T
    parser.add_argument("--document_repr_type", default="mixture-100")
    parser.add_argument("--attention_mechanism", type=str, default="mixture")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--num_expected", type=int, default=9)

    args = parser.parse_args()
    print(vars(args))
    main(args.dataset_name, args.pca, args.cluster_method, args.rep_type, args.lm_type, args.document_repr_type, args.attention_mechanism, args.layer, args.random_state, args.num_expected)
