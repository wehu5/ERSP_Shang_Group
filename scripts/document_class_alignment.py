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

#Partitions document representations into low and high confidence groups
def partitionDataset(threshold, doc_reps, known_class_reps):
    #Matrix of cosine similarities with respect to known class representations
    cosine_similarities = cosine_similarity_embeddings(doc_reps, known_class_reps) 
    document_class_assignment = np.argmax(cosine_similarities, axis=1) #Cosine similarity predictions
    low_conf_docs = []
    high_conf_docs = []
    #Closest of the known classes based on similarity between doc_rep and known_class_reps.
    #Note: uses indices relative to known_class_reps, so the indices don't make sense when
    #referring to all original classes.    
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
    save_dict_data = {}

    # pca = 0 means no pca
    do_pca = pca != 0

    save_dict_data["dataset_name"] = dataset_name
    save_dict_data["pca"] = pca
    save_dict_data["cluster_method"] = cluster_method
    save_dict_data["lm_type"] = lm_type
    save_dict_data["document_repr_type"] = document_repr_type
    save_dict_data["random_state"] = random_state

    naming_suffix = f"pca{pca}.clus{cluster_method}.{lm_type}-{layer}.{document_repr_type}.{random_state}"
    print(f"naming_suffix: {naming_suffix}")

    data_dir = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name)
    print(f"data_dir: {data_dir}")

    with open(os.path.join(data_dir, "dataset.pk"), "rb") as f:
        dictionary = pk.load(f)
        class_names = dictionary["class_names"]
        num_classes = len(class_names)
        print(class_names)

    with open(os.path.join(data_dir, f"tokenization_lm-{lm_type}-{layer}.pk"), "rb") as f:
        tokenization_info = pk.load(f)["tokenization_info"]

    # huh?
    with open(os.path.join(data_dir, f"document_repr_lm-{lm_type}-{layer}-{document_repr_type}.pk"), "rb") as f:
        dictionary = pk.load(f)
        document_representations = dictionary["document_representations"]
        class_representations = dictionary["class_representations"]
        raw_document_representations = dictionary["raw_document_representations"]
        repr_prediction = np.argmax(cosine_similarity_embeddings(document_representations, class_representations),
                                    axis=1)
        save_dict_data["repr_prediction"] = repr_prediction


    if do_pca:
        _pca = PCA(n_components=pca, random_state=random_state)
        document_representations = _pca.fit_transform(document_representations)
        raw_document_representations = _pca.transform(raw_document_representations)
        class_representations = _pca.transform(class_representations)
        print(f"Explained variance: {sum(_pca.explained_variance_ratio_)}")

    # partition dataset
    low_conf_docs, high_conf_docs, document_class_assignment = partitionDataset(0.10, document_representations, class_representations)

    # cluster low-confidence documents
    low_conf_doc_reps = replace_with_raw(low_conf_docs, raw_document_representations)
    kmeans = KMeans(n_clusters=num_expected, random_state=random_state, init='k-means++')
    kmeans.fit(low_conf_doc_reps)

    # Get kmeans predictions, cluster centers
    low_conf_doc_predictions = kmeans.predict(low_conf_doc_reps)
    
    #  do we need this? idk
    low_conf_centers = kmeans.cluster_centers_

    # keyword generation
    # ->
    # Grab indices with respect to all documents of low_conf_docs
    low_conf_indices = [ doc_tuple[1] for doc_tuple in low_conf_docs ]
    cluster_keywords = generate_keywords(tokenization_info, low_conf_doc_predictions, low_conf_indices, num_expected)

    for keywords in cluster_keywords:
        print("Cluster Words :")
        print(keywords)

    return

    # class representations building 
    # -> final_class_representations

    # MUST DO BELOW
    # final_class_representations = np.array(final_class_representations)

    if rep_type == 'generated':

        # put together new document representations from all documents
        # these are class aligned with the new classes
        # -> final_doc_representations

        # final_doc_representations = generate_doc_representations(final_class_representations, attention_mechanism, lm_type, layer)

        if do_pca:
            _pca = PCA(n_components=pca, random_state=random_state)
            final_doc_representations = _pca.fit_transform(final_doc_representations)
            final_class_representations = _pca.transform(final_class_representations)
            print(f"Final explained variance: {sum(_pca.explained_variance_ratio_)}")
        
    elif rep_type == "raw":

        # put together document representations from high + raw from low, excluding ill performing
        # -> final_doc_representations
        print("remove later")

    if cluster_method == 'gmm':

        cosine_similarities = cosine_similarity_embeddings(final_doc_representations, final_class_representations)
        document_class_assignment = np.argmax(cosine_similarities, axis=1)
        document_class_assignment_matrix = np.zeros((final_doc_representations.shape[0], num_expected))
        for i in range(final_doc_representations.shape[0]):
            document_class_assignment_matrix[i][document_class_assignment[i]] = 1.0

        gmm = GaussianMixture(n_components=num_expected, covariance_type='tied',
                              random_state=random_state,
                              n_init=999, warm_start=True)
        gmm.converged_ = "HACK"

        gmm._initialize(final_doc_representations, document_class_assignment_matrix)
        gmm.lower_bound_ = -np.infty
        gmm.fit(final_doc_representations)

        documents_to_class = gmm.predict(final_doc_representations)
        centers = gmm.means_
        save_dict_data["centers"] = centers
        distance = -gmm.predict_proba(final_doc_representations) + 1

    elif cluster_method == 'kmeans':

        kmeans = KMeans(n_clusters=num_expected, init=final_class_representations, random_state=random_state)
        kmeans.fit(final_doc_representations)

        documents_to_class = kmeans.predict(final_doc_representations)
        centers = kmeans.cluster_centers_
        save_dict_data["centers"] = centers
        distance = np.zeros((final_doc_representations.shape[0], centers.shape[0]), dtype=float)
        for i, _emb_a in enumerate(final_doc_representations):
            for j, _emb_b in enumerate(centers):
                distance[i][j] = np.linalg.norm(_emb_a - _emb_b)

    save_dict_data["documents_to_class"] = documents_to_class
    save_dict_data["distance"] = distance

    with open(os.path.join(data_dir, f"data.{naming_suffix}.pk"), "wb") as f:
        pk.dump(save_dict_data, f)


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
