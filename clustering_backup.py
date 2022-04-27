#Original XClass imports
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

#For creating confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


#Displays a confusion matrix for clustering results vs ground truth 
def plotConfusionMatrix(data_dir, predictions, dataset_name, orig_class_names, threshold, mask, cluster_method):
    #Import all documents' ground truth labels
    labels_path = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, f"../datasets/{dataset_name}", "labels.txt")
    with open(labels_path, "r") as f:
        lines = np.loadtxt(f)
        labels = np.array(lines)

    #Calculate confusion matrix using ground truth vs cluster predictions
    conf_matrix = confusion_matrix(labels, predictions, labels=list(range(len(orig_class_names)))) #Can we use labels=classes?
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=orig_class_names)
    #Display and save the figure
    disp.plot()
    plt.title(f"{args.dataset_name} threshold={threshold}, mask={args.mask}, cluster={args.cluster_method}")
    plt.xticks(rotation=45)
    # plt.show()
    save_path = f"../tsne_output/Clustering experiments (malcolm)/hybrid clustering/kmeansx1gmmx1_maskingFixed_{dataset_name}_{threshold},{mask},{cluster_method}.png"
    plt.savefig(save_path, bbox_inches='tight')
   

#Imports files related to the dataset
def importDataset(dataset_name,
                  pca,
                  cluster_method,
                  lm_type,
                  document_repr_type,
                  random_state,
                  mask,
                  save_dict_data,
                  data_dir):

    #dataset.pk
    with open(os.path.join(data_dir, "dataset.pk"), "rb") as f:
        dataset_pk = pk.load(f)
        datasetpk_class_names = dataset_pk["class_names"]      

    #Original class names
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, f"../datasets/{dataset_name}", "original_classes.txt"), "r") as f:
        orig_class_names = f.readlines()

    #Known class names (the class names remaining after masking)
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, f"../datasets/{dataset_name}", f"classes_{mask}.txt"), "r") as f:
        known_class_names = f.readlines()
        print(f"known classes = {known_class_names}")     

    #Document and class representations
    with open(os.path.join(data_dir, "document_representations", f"document_repr_lm-{lm_type}-{document_repr_type}_{mask}.pk"), "rb") as f:
        document_representations_dict = pk.load(f)
        doc_reps = document_representations_dict["document_representations"]
        known_class_reps = document_representations_dict["class_representations"]
        repr_prediction = np.argmax(cosine_similarity_embeddings(doc_reps, known_class_reps), axis=1)
        save_dict_data["repr_prediction"] = repr_prediction

    #Strip newline from class names
    for i,name in enumerate(known_class_names):
        known_class_names[i] = name.rstrip('\n')
    for i,name in enumerate(orig_class_names):
        orig_class_names[i] = name.rstrip('\n')

    #Checking imported file dimensions/sizes
    print(f"Number of datasetpk_class_names = {len(datasetpk_class_names)}")
    print(f"Number of doc_reps = {len(doc_reps)}")
    print(f"Number of orig_class_names = {len(orig_class_names)}")
    
    print(f"Number of known classes = {len(known_class_names)}")
    print(f"Number of known_class_reps = {len(known_class_reps)}")
    print(f"known_class_names = {known_class_names}")
    print(f"orig_class_names = {orig_class_names}")

    #Store all imported values in import_dict, and return it
    import_dict = {}    
    import_dict["datasetpk_class_names"] = datasetpk_class_names
    import_dict["orig_class_names"]      = orig_class_names
    import_dict["known_class_names"]     = known_class_names
    import_dict["doc_reps"]              = doc_reps
    import_dict["known_class_reps"]      = known_class_reps
    import_dict["repr_prediction"]       = repr_prediction 
    return import_dict


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
    print(f"Number of total docs = {len(doc_reps)}, sum of low+high = {len(low_conf_docs) + len(high_conf_docs)}")

    return low_conf_docs, high_conf_docs, document_class_assignment


def main(dataset_name,
         pca,
         cluster_method,
         lm_type,
         document_repr_type,
         random_state,
         mask):

    # pca = 0 means no pca
    do_pca = pca != 0

    save_dict_data = {}
    save_dict_data["dataset_name"] = dataset_name
    save_dict_data["pca"] = pca
    save_dict_data["cluster_method"] = cluster_method
    save_dict_data["lm_type"] = lm_type
    save_dict_data["document_repr_type"] = document_repr_type
    save_dict_data["random_state"] = random_state

    naming_suffix = f"pca{pca}.clus{cluster_method}.{lm_type}.{document_repr_type}.{random_state}"
    print(f"naming_suffix: {naming_suffix}")

    data_dir = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name)
    print(f"data_dir: {data_dir}")

    #Unpack imported data
    import_dict = importDataset(dataset_name, pca, cluster_method, lm_type,
                                document_repr_type, random_state, mask, save_dict_data, data_dir)
    datasetpk_class_names = import_dict["datasetpk_class_names"]
    orig_class_names      = import_dict["orig_class_names"]
    known_class_names     = import_dict["known_class_names"]
    doc_reps              = import_dict["doc_reps"]
    known_class_reps      = import_dict["known_class_reps"]
    repr_prediction       = import_dict["repr_prediction"]

    #Apply PCA
    if do_pca:
        _pca = PCA(n_components=pca, random_state=random_state)
        doc_reps = _pca.fit_transform(doc_reps)
        known_class_reps = _pca.transform(known_class_reps)
        print(f"Explained variance: {sum(_pca.explained_variance_ratio_)}")


    #Start of clustering code------------------------------------------------------------
    #Try partitioning and clustering low-confidence documents on different thresholds
        # confidence_thresholds = [ 0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.15, 0.20 ]
    # confidence_thresholds = [ 0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80 ]
    # confidence_thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.5, 0.6, 0.7, 0.8]
    # confidence_thresholds = [ 0.15 ]
    confidence_thresholds = [0.01,0.02,0.04,0.06,0.08,0.10]
    for threshold in confidence_thresholds:
        #Partition dataset
        low_conf_docs, high_conf_docs, document_class_assignment = partitionDataset(threshold, doc_reps, known_class_reps)

        #Cluster low-confidence documents
        num_unknown_classes = len(orig_class_names) - len(known_class_names)
        kmeans = KMeans(n_clusters=num_unknown_classes, random_state=random_state)
        low_conf_doc_reps = [doc_tuple[0] for doc_tuple in low_conf_docs] #Get only the document representation from the tuple
        kmeans.fit(low_conf_doc_reps)

        low_conf_doc_predictions = kmeans.predict(low_conf_doc_reps)
            # print(f"low_conf_doc_predictions shape = {low_conf_doc_predictions.shape}")
        low_conf_centers = kmeans.cluster_centers_
            #What is shape of low_conf_centers?
        print(f"low_conf_centers.shape = {low_conf_centers.shape}")
            # save_dict_data["centers"] = centers
            # unknown_class_distances = np.zeros((len(low_conf_docs), low_conf_centers.shape[0]), dtype=float)

            # # Distance of every low confidence document representations from each other?
            # for i, _emb_a in enumerate(low_conf_docs):
            #     for j, _emb_b in enumerate(low_conf_centers):
            #         unknown_class_distances[i][j] = np.linalg.norm(_emb_a - _emb_b)

        ''' Penultimate kmeans
        # Penultimate kmeans using all document representations, initialized with low confidence cluster centers
        # and known class representations.
        # Goal is to fix any imbalance in the confidence partitioning step.
        # hybrid_centers should be list of known class representations, followed by low confidence clustering centers
        hybrid_centers = np.concatenate( (known_class_reps,low_conf_centers), axis=0 ) 
        print(f"hybrid_centers.shape = {hybrid_centers.shape}")
        kmeans = KMeans(n_clusters=len(orig_class_names), init=hybrid_centers, random_state=random_state)
        kmeans.fit(doc_reps)

        final_predictions = kmeans.predict(doc_reps)
        centers = kmeans.cluster_centers_
        save_dict_data["centers"] = centers
            # distance = np.zeros((document_representations.shape[0], centers.shape[0]), dtype=float)
            # for i, _emb_a in enumerate(document_representations):
            #     for j, _emb_b in enumerate(centers):
            #         distance[i][j] = np.linalg.norm(_emb_a - _emb_b)
        '''

        #Final GMM clustering with (# clusters) = (#known classes) + (#unknown classes)
        if cluster_method == 'gmm':
            # kmeansx1, gmmx1
            document_class_assignment_matrix = np.zeros((doc_reps.shape[0], len(orig_class_names)))
            #High confidence documents use known classes
            for doc_tuple in high_conf_docs:
                doc_rep = doc_tuple[0]
                doc_index = doc_tuple[1]
                cosine_prediction = document_class_assignment[doc_index]
                if cosine_prediction < 0 or cosine_prediction > 3:
                    print("cosine_prediction indexing issue")              
                document_class_assignment_matrix[doc_index][cosine_prediction] = 1.0 #[specifices which document][gives known class label]
            #Low confidence documents use closest unknown cluster center
            for i,doc_tuple in enumerate(low_conf_docs):
                doc_rep = doc_tuple[0]
                doc_index = doc_tuple[1]                
                # Zero-indexed relative to only unknown classes, shift index to account for known classes
                kmeans_prediction = low_conf_doc_predictions[i] + len(known_class_names)
                if kmeans_prediction < 4 or kmeans_prediction > 8:
                    print("kmeans_prediction indexing issue")
                document_class_assignment_matrix[doc_index][kmeans_prediction] = 1.0 



            ''' Penultimate kmeans
            # GMM initialization uses penultimate kmeans clustering predictions.
            # document_class_assignment_matrix dimensions are (#docs)x(#original classes)
            document_class_assignment_matrix = np.zeros((doc_reps.shape[0], len(orig_class_names)))
            # Loop through predictions for all documents and initialize matrix
            for i,prediction in enumerate(final_predictions):
                document_class_assignment_matrix[i][prediction] = 1
            '''

            gmm = GaussianMixture(n_components=len(orig_class_names), covariance_type='tied',
                                  random_state=random_state,
                                  n_init=999, warm_start=True)
            gmm.converged_ = "HACK"

            gmm._initialize(doc_reps, document_class_assignment_matrix)
            gmm.lower_bound_ = -np.infty
            gmm.fit(doc_reps)

            final_predictions = gmm.predict(doc_reps)
            centers = gmm.means_
            save_dict_data["centers"] = centers
            distance = -gmm.predict_proba(doc_reps) + 1
            

                #Save to pickle file
                # save_dict_data["documents_to_class"] = documents_to_class
                # save_dict_data["distance"] = distance

                # # with open(os.path.join(data_dir, f"data.{naming_suffix}_{args.mask}.pk"), "wb") as f:
                # #     pk.dump(save_dict_data, f)

        save_dict_data["documents_to_class"] = final_predictions
        save_dict_data["distance"] = distance

        with open(os.path.join(data_dir, f"data.{naming_suffix}_{args.mask}_{threshold}.pk"), "wb") as f:
            pk.dump(save_dict_data, f)

        plotConfusionMatrix(data_dir, final_predictions, dataset_name,
                            orig_class_names, threshold, mask, cluster_method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="NYT_Topics")
    parser.add_argument("--pca", type=int, default=64, 
                        help="number of dimensions projected to in PCA, "
                             "-1 means not doing PCA.")
    parser.add_argument("--cluster_method", choices=["gmm", "kmeans"],
                        default="gmm")
    # language model + layer
    parser.add_argument("--lm_type", default="bbu-12")
    # attention mechanism + T
    parser.add_argument("--document_repr_type", default="mixture-100")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--mask", type=str, default="50%")

    args = parser.parse_args()
    print(f"vars(args): {vars(args)}")
    main(args.dataset_name, args.pca, args.cluster_method, args.lm_type, 
         args.document_repr_type, args.random_state, args.mask)









# -------------Previous GMM initialization before adding the penultimate kmeans----------
            #GMM initialization
                # closest_low_confidence_cluster = np.argmax(unknown_class_distances, axis=1) #zero indexing
            #Matrix of zeroes, dimensions are (#docs)x(#original classes)
            # document_class_assignment_matrix = np.zeros((doc_reps.shape[0], len(orig_class_names)))
            # #High confidence documents use known classes
            # for doc_tuple in high_conf_docs:
            #     # j = high_conf_indices[i]
            #     doc_rep = doc_tuple[0]
            #     doc_index = doc_tuple[1]
            #     cosine_predicted_class = document_class_assignment[doc_index]
            #     converted_label = labelKnownToOriginal(allClassDict,knownDict,cosine_predicted_class)
            #     # converted_label = labelKnownToOriginal(class_name_indices_dict,
            #     #                                        known_class_label_to_name_dict,
            #     #                                        cosine_predicted_class) #Converts label relative to known classes to label relative to original classes
            #     document_class_assignment_matrix[doc_index][converted_label] = 1.0 #[specifices which document][gives known class label]
            # #Low confidence documents use closest unknown cluster center
            # for i,doc_tuple in enumerate(low_conf_docs):
            #     doc_rep = doc_tuple[0]
            #     doc_index = doc_tuple[1]                
            #     kmeans_prediction = low_conf_doc_predictions[i] #closest_low_confidence_cluster[i]
            #     converted_label = labelUnknownToOriginal(allClassDict,unknownDict,kmeans_prediction)
            #     # converted_label = labelUnknownToOriginal(class_name_indices_dict,
            #     #                                          unknown_class_label_to_name_dict,
            #     #                                          kmeans_prediction)
            #     # j = low_conf_indices[i]
            #     document_class_assignment_matrix[doc_index][converted_label] = 1.0 

# -------------Aligning unknown classes with original indices, obsolete-----------------
    # #Dictionary to convert known or unknown class name to its original index in the original classes.txt
    # class_name_indices_dict = {}
    # # If it's a known name, we can easily find it in originalclasses.txt
    # for i,orig_name in enumerate(orig_class_names):
    #     for known_name in known_class_names:
    #         if known_name == orig_name:
    #             class_name_indices_dict[known_name] = i

    # for i,orig_name in enumerate(orig_class_names):
    #     if orig_name not in masked_to_orig_indices:
    #         masked_to_orig_indices[orig_name] = i
    # #Now we can convert any class name to its proper index.

    # #Dictionary to convert known class integer label to class name
    # known_class_label_to_name_dict = {}
    # for i,known_class in enumerate(known_class_names):
    #     known_class_label_to_name_dict[i] = known_class

    # # Dictionary to convert unknown class integer label to class name --> Problem!
    # # How to figure out which kmeans cluster is which unknown class name?
    # unknown_class_label_to_name_dict = {}
    # for i,unknown_class in enumerate(unknown_class_names)


# # def labelKnownToOriginal(class_name_indices_dict, known_class_label_to_name_dict, label):
# def labelKnownToOriginal(allClassDict, knownDict, label):
#     return knownDict[label]
#     # class_name = known_class_label_to_name_dict[label]
#     # corrected_label = class_name_indices_dict[class_name]
#     # return corrected_label


# # def labelUnknownToOriginal(class_name_indices_dict, unknown_class_label_to_name_dict, label):
# def labelUnknownToOriginal(allClassDict, unknownDict, label):
#     return unknownDict[label]
#     # # print(f"label = {label}")
#     # class_name = unknown_class_label_to_name_dict[label]
#     # # class_name = class_name.rstrip('\n')
#     # corrected_label = class_name_indices_dict[class_name]
#     # return corrected_label

    # #Dictionary to convert known or unknown class name to its original index in the original classes.txt
    # class_name_indices_dict = {}
    # # If it's a known name, we can easily find it in originalclasses.txt
    # for i,orig_name in enumerate(orig_class_names):
    #     for known_name in known_class_names:
    #         if known_name == orig_name:
    #             class_name_indices_dict[known_name] = i

    # unknown_class_names = []
    # for i,orig_name in enumerate(orig_class_names):
    #     if orig_name not in class_name_indices_dict:
    #         unknown_class_names.append(orig_name)
    #         # masked_to_orig_indices[orig_name] = i
    # #Now we can convert any class name to its proper index.
    # print(unknown_class_names)
    # #Dictionary to convert known class integer label to class name.
    # known_class_label_to_name_dict = {}
    # for i,known_class in enumerate(known_class_names):
    #     known_class_label_to_name_dict[i] = known_class

    # # Dictionary to convert unknown class integer label to class name --> Problem!
    # # How to figure out which kmeans cluster is which unknown class name?
    # # Currently not aligned!
    # unknown_class_label_to_name_dict = {}
    # for i,unknown_class in enumerate(unknown_class_names):
    #     unknown_class_label_to_name_dict[i] = unknown_class

    # for i,orig_name in enumerate(orig_class_names):
    #     for unknown_name in unknown_class_names:
    #         if unknown_name == orig_name:
    #             class_name_indices_dict[unknown_name] = i    

    # unknown_class_names = []
    # for name in orig_class_names:
    #     isKnown = False
    #     for known in known_class_names:
    #         if name == known:
    #             isKnown = True
    #     if isKnown == False:
    #         unknown_class_names.append(name)


    # allClassDict = {}
    # for i,name in enumerate(orig_class_names):
    #     allClassDict[name] = i

    # knownDict = {}
    # for i,name in enumerate(known_class_names):
    #     for j,orig in enumerate(orig_class_names):
    #         if name == orig:
    #             knownDict[i] = j

    # unknownDict = {}
    # for i,name in enumerate(unknown_class_names):
    #     for j,orig in enumerate(orig_class_names):
    #         if name == orig:
    #             unknownDict[i] = j


    # # print(class_name_indices_dict)
    # # print(known_class_label_to_name_dict)
    # # print(unknown_class_label_to_name_dict)
    # # indices_fix = {}
    # # for i,orig_name in enumerate(orig_class_names):
    # #     if orig_name in 






# --------------------JUNK-------------------------------------------------
    # for i in range(document_representations.shape[0]):
    #     if cosine_similarities[i][document_class_assignment[i]] >= confidence_threshold:
    #         document_class_assignment_matrix[i][document_class_assignment[i]] = 1.0    
    #     else:
    #         closest_cluster = np.argmax(d)
    #         document_class_assignment_matrix[i][closest_low_confidence_cluster[]]
