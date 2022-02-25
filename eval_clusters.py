import argparse
import math
import os
import pickle as pk
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def main(args):

    '''
        loads the clustered data from the gmm pickle file and obtain the predicted labels of the documents \n
        Stores in the doc_to_class variable
    '''
    cluster_data_dir = f"../data/intermediate_data/{args.dataset_name}/data.pca64.clusgmm.bbu-12.mixture-100.42.pk"
    with open(cluster_data_dir, "rb") as f:
        cluster_data = pk.load(f)
        doc_to_class = cluster_data["documents_to_class"]

    '''
        from the dataset that corresponds to the input dataset name, obtain the labels and store them in the 'label' array
    '''
    dataset_path = os.path.expanduser("~/XClass/data/datasets/")
    with open(os.path.join(dataset_path, args.dataset_name, "labels.txt"), "r") as f:
        lines = f.readlines()
    label = np.array(lines) # I manually checked the length of doc_to_class using Malcolm's script, 
                            # its length is the same as the number of labels(the length of the file) in the labels.txt file

    '''
        reads the classes file to find the number of labels(classes) to use in constructing the confusion matrix
    '''
    with open(os.path.join(dataset_path, args.dataset_name, "original_classes.txt"), "r") as f:
        classes = f.readlines()
    num_labels = len(classes)

    '''
        'con_matrx' is the confusion matrix
        arguments: confusion_matrix(y1, y2, labels)
        y1 = the true labels
        y2 = the predicted labels by gmm
        labels = the labels in the dataset to label the rows &  columns of the confusion matrix, actually
            instead of using the number labels, we could use 'labels=classes' here to use the class names
        there are other parameters for this function but in this case I only used these three
        'con_matrx' should be a n by n matrix where n is the number of classes
    '''
    con_matrx = confusion_matrix(label, doc_to_class, labels=list(range(num_labels)))
    print(con_matrx)

    '''
        I also stores the true/predicted labels in the dataframe df in case we want to calculate something else.
        Below I simply calculated the accuracy of the gmm prediction by comparing the two Series index by index
        the columns of the dataframe are named 'true_labels' and 'gmm_predictions'.
    '''
    df = pd.DataFrame(np.array(label, doc_to_class), columns=['true_labels', 'gmm_predictions'])
    percent_correct = ((df['true_labels'] == df['gmm_predictions']).sum() / df.shape[0]) * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str)
    args = parser.parse_args()
    print(args)
    print(vars(args))
    main(args)