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
from scipy.optimize import linear_sum_assignment as hungarian


#Displays a confusion matrix for clustering results vs ground truth 
def plotConfusionMatrix(predictions, low_conf_docs, dataset_name, orig_class_names, threshold, mask, cluster_method):
    #Import all documents' ground truth labels
    labels_path = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, f"../datasets/{dataset_name}", "labels.txt")
    with open(labels_path, "r") as f:
        lines = np.loadtxt(f)
        labels = np.array(lines)

    low_conf_labels = []
    for doc_tuple in low_conf_docs:
        doc_index = doc_tuple[1] # Relative to all documents
        low_conf_labels.append(labels[doc_index])

    # Calculate confusion matrix using ground truth vs cluster predictions and cosine predictions
    conf_matrix = confusion_matrix(low_conf_labels, predictions, labels=list(range(len(orig_class_names)))) #Can we use labels=classes?
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=orig_class_names)
    #Display and save the figure
    disp.plot()
    plt.title(f"{args.dataset_name} threshold={threshold}, mask={args.mask}, kmeans++, known class low-conf clusters")
    plt.xticks(rotation=45)
    plt.show()
    # save_path = f"../tsne_output/Clustering experiments/lowconfknownclassclusters/{dataset_name}_{threshold},{mask}, kmeans++.png"
    # plt.savefig(save_path, bbox_inches='tight')

   

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

    data_dir_class_oriented = os.path.join(data_dir, "NYT-Topics_50%_classOriented_classesOrdered")
    data_dir_raw = os.path.join(data_dir, "NYT-Topics_50%_raw_classesOrdered")

    #dataset.pk
    with open(os.path.join(data_dir_class_oriented, "dataset.pk"), "rb") as f:
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
    # Usual path = os.path.join(data_dir, "document_representations", f"document_repr_lm-{lm_type}-{document_repr_type}_{mask}.pk")
    nytTopics_4_27_22 = os.path.join(data_dir_class_oriented, f"document_repr_lm-{lm_type}-{document_repr_type}.pk")
    with open(nytTopics_4_27_22, "rb") as f:
        document_representations_dict = pk.load(f)
        doc_reps = document_representations_dict["document_representations"]
        known_class_reps = document_representations_dict["class_representations"]
        repr_prediction = np.argmax(cosine_similarity_embeddings(doc_reps, known_class_reps), axis=1)
        save_dict_data["repr_prediction"] = repr_prediction

    nytTopics_raw_5_3_2022 = os.path.join(data_dir_raw, f"document_repr_lm-{lm_type}-{document_repr_type}.pk")
    with open(nytTopics_raw_5_3_2022, "rb") as f:
        raw_document_representations_dict = pk.load(f)
        raw_doc_reps = raw_document_representations_dict["document_representations"]

    #Strip newline from class names
    for i,name in enumerate(known_class_names):
        known_class_names[i] = name.rstrip('\n')
    for i,name in enumerate(orig_class_names):
        orig_class_names[i] = name.rstrip('\n')

    # #Checking imported file dimensions/sizes
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
    import_dict["raw_doc_reps"]          = raw_doc_reps
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

    return low_conf_docs, high_conf_docs, document_class_assignment

def evalCos(doc_reps, known_class_reps, dataset_name, orig_class_names):

    #Import all documents' ground truth labels
    labels_path = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, f"../datasets/{dataset_name}", "labels.txt")
    with open(labels_path, "r") as f:
        lines = np.loadtxt(f)
        labels = np.array(lines)

    cosine_similarities = cosine_similarity_embeddings(doc_reps, known_class_reps) 
    cosine_similarities = np.abs(cosine_similarities)
    document_class_assignment = np.argmax(cosine_similarities, axis=1) #Cosine similarity predictions
   
    class_dict = [[] for class_name in orig_class_names]

    for i, doc_rep in enumerate(doc_reps):       
        doc_max_sim = cosine_similarities[i][document_class_assignment[i]]
        class_dict[int(labels[i])].append(doc_max_sim)

    hist_bins = [round(-1 + i*0.1,1) for i in range(21)]
    xticklabels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    # print(f"hist_bins = {hist_bins}")
    fig, axes = plt.subplots(3,3,figsize=(15,15), constrained_layout=True)
    for index,class_name in enumerate(orig_class_names):
        class_values = class_dict[index] # List of max cosine similarities of docs in class_name        
        axes.flat[index].set_title(f"'{class_name}'")
        axes.flat[index].set_xticks(np.array(xticklabels)) # hist_bins
        axes.flat[index].hist(class_values, bins=hist_bins)

    fig.suptitle("Cosine similarities for {class-oriented, NYT-Topics, 50% masking}")
    plt.show()


def mergeCluster(centers, center_classrep_distances, known_class_reps, orig_class_names):
    # Assign which low-confidence clusters will be absorbed by known classes, using Hungarian matching algorithm
    row_ind,col_ind = hungarian(center_classrep_distances)
    print(f"row_ind = {row_ind}")
    print(f"col_ind = {col_ind}")
    cluster_to_classes = {}
    for i,row in enumerate(row_ind):
        cluster_to_classes[row] = col_ind[i]
    unknown_class_counter = len(known_class_reps) # This line assumes known classes come first uninterrupted in classes.txt
    for i in range(len(orig_class_names)):
        if i in cluster_to_classes:
            continue
        else:
            cluster_to_classes[i] = unknown_class_counter
            unknown_class_counter += 1
    print(f"cluster_to_classes = \n{cluster_to_classes}")    
    return cluster_to_classes

def mergeClusterCosine(centers, known_class_reps, orig_class_names):
    # Assign which low-confidence clusters will be absorbed by known classes, using Hungarian matching algorithm
    center_classrep_similarity = cosine_similarity_embeddings(centers, known_class_reps)
    print(f"center_classrep_similarity shape = {center_classrep_similarity.shape}")
    print(center_classrep_similarity)
    row_ind,col_ind = hungarian(center_classrep_similarity, maximize=True)
    print(f"row_ind = {row_ind}")
    print(f"col_ind = {col_ind}")
    cluster_to_classes = {}
    for i,row in enumerate(row_ind):
        cluster_to_classes[row] = col_ind[i]
    unknown_class_counter = len(known_class_reps) # This line assumes known classes come first uninterrupted in classes.txt
    for i in range(len(orig_class_names)):
        if i in cluster_to_classes:
            continue
        else:
            cluster_to_classes[i] = unknown_class_counter
            unknown_class_counter += 1
    print(f"cluster_to_classes = \n{cluster_to_classes}")    
    return cluster_to_classes    

def replaceWithRaw(low_conf_docs, raw_doc_reps):
    raw_low_conf_docs = []
    
    for doc, index in low_conf_docs:
        raw_low_conf_docs.append(raw_doc_reps[index])

    return raw_low_conf_docs

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
    raw_doc_reps          = import_dict["raw_doc_reps"]
    known_class_reps      = import_dict["known_class_reps"]
    repr_prediction       = import_dict["repr_prediction"]
    
    #Apply PCA
    if do_pca:
        _pca = PCA(n_components=pca, random_state=random_state)
        doc_reps = _pca.fit_transform(doc_reps)
        raw_doc_reps = _pca.fit_transform(raw_doc_reps)
        known_class_reps = _pca.transform(known_class_reps)
        print(f"Explained variance: {sum(_pca.explained_variance_ratio_)}")

    print (f"raw_doc_rep.shape = {raw_doc_reps.shape}")

#Start of clustering code------------------------------------------------------------
    #Try partitioning and clustering low-confidence documents on different thresholds
    confidence_thresholds = [0.10, 0.15, 0.20] #[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    confidence_thresholds = [0.01, 0.03, 0.05, 0.10, 0.15]
    for threshold in confidence_thresholds:
        # (FOR EVALUATIONS) evalCos(doc_reps, known_class_reps, dataset_name, orig_class_names)

        #Partition dataset
        low_conf_docs, high_conf_docs, document_class_assignment = partitionDataset(threshold, doc_reps, known_class_reps)

        #Cluster low-confidence documents
        # ( NO RAW DOCUMENT REPLACEMENT) low_conf_doc_reps = [doc_tuple[0] for doc_tuple in low_conf_docs] #Get only the document representation from the tuple
        low_conf_doc_reps = replaceWithRaw(low_conf_docs, raw_doc_reps)
        kmeans = KMeans(n_clusters=len(orig_class_names), random_state=random_state, init='k-means++')
        kmeans.fit(low_conf_doc_reps)

        # Get kmeans predictions, cluster centers
        low_conf_doc_predictions = kmeans.predict(low_conf_doc_reps)
        low_conf_centers = kmeans.cluster_centers_
        print(f"low_conf_doc_predictions shape = {low_conf_doc_predictions.shape}")
        print(f"low_conf_centers.shape = {low_conf_centers.shape}")
        save_dict_data["centers"] = low_conf_centers

        plotConfusionMatrix(low_conf_doc_predictions, low_conf_docs, dataset_name,
                            orig_class_names, threshold, mask, cluster_method)

        # Distance of low-confidence cluster centers from known class representations
#        center_classrep_distances = np.zeros((len(orig_class_names), len(known_class_names)), dtype=float) # (num original classes)x(num known classes)
#        for i,center in enumerate(low_conf_centers):
#            for j,class_rep in enumerate(known_class_reps):
#                center_classrep_distances[i][j] = np.linalg.norm(center - class_rep)
#        print(f"center to class rep distances shape = {center_classrep_distances.shape}")
#        print(center_classrep_distances)

        # cluster_to_classes = mergeCluster(low_conf_centers, center_classrep_distances, known_class_reps, orig_class_names)
        cluster_to_classes = mergeClusterCosine(low_conf_centers, known_class_reps, orig_class_names)

        # Assign final_predictions with low-confidence kmeans predictions and Hungarian matching
        final_predictions = [-1 for i in range(len(doc_reps))]
        for i,doc_tuple in enumerate(low_conf_docs):
            kmeans_prediction = low_conf_doc_predictions[i]
            all_docs_index = low_conf_docs[i][1] # Index w.r.t. all doc_reps
            final_predictions[all_docs_index] = cluster_to_classes[kmeans_prediction]
                # if kmeans_prediction in row_ind:
                #     # Low-conf document absorbed by known classes
                #     known_class_prediction = cluster_to_classes[kmeans_prediction]
                #     final_predictions[all_docs_index] = known_class_prediction
                # else:
        #High confidence documents use known classes
        for doc_tuple in high_conf_docs:
            doc_rep = doc_tuple[0]
            doc_index = doc_tuple[1]
            cosine_prediction = document_class_assignment[doc_index]
            if cosine_prediction < 0 or cosine_prediction > 3:
                print("cosine_prediction indexing issue")     
            if final_predictions[doc_index] != -1:
                 print("final_prediction being overwritten") 
            final_predictions[doc_index] = cosine_prediction                    

        # plotConfusionMatrix(final_predictions, dataset_name,
        #                     orig_class_names, threshold, mask, cluster_method)

        # save_dict_data["documents_to_class"] = final_predictions
        # save_dict_data["distance"] = distance
        # with open(os.path.join(data_dir, f"data.{naming_suffix}_{args.mask}_{threshold}_.pk"), "wb") as f:
        #     pk.dump(save_dict_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="NYT-Topics")
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