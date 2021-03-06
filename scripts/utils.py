import itertools
import operator
import os

import numpy as np

linewidth = 200
np.set_printoptions(linewidth=linewidth)
np.set_printoptions(precision=3, suppress=True)

from collections import Counter

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertModel, BertTokenizer

MODELS = {
    'bbc': (BertModel, BertTokenizer, 'bert-base-cased'),
    'bbu': (BertModel, BertTokenizer, 'bert-base-uncased')
}

# all paths can be either absolute or relative to this utils file
DATA_FOLDER_PATH = os.path.join('..', 'data', 'datasets')
INTERMEDIATE_DATA_FOLDER_PATH = os.path.join('..', 'data', 'intermediate_data')
# this is also defined in run_train_text_classifier.sh, make sure to change both when changing.
FINETUNE_MODEL_PATH = os.path.join('..', 'models')


def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


def dot_product_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b))


def cosine_similarity_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)


def pairwise_distances(x, y):
    return cdist(x, y, 'euclidean')


def most_common(L):
    c = Counter(L)
    return c.most_common(1)[0][0]


def evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False):
    
    confusion = confusion_matrix(true_class, predicted_class)
    # Manually swapping columns for evaluation
    confusion[:, [4,6]] = confusion[:, [6,4]]
    confusion[:, [5,6]] = confusion[:, [6,5]]
    # Change prediction labels
    import copy
    predicted_class_swapped = copy.deepcopy(predicted_class)
    for i in range(len(predicted_class)):
        # Education --> Estate
        if predicted_class[i] == 4:
            predicted_class_swapped[i] = 5
        # Estate --> Arts
        if predicted_class[i] == 5:
            predicted_class_swapped[i] = 6
        # Arts --> Education
        if predicted_class[i] == 6:
            predicted_class_swapped[i] = 4
    print(predicted_class[:100])
    print(predicted_class_swapped[:100])
    if output_to_console:
        print("-" * 80 + "Evaluating" + "-" * 80)
        print(confusion)
    f1_macro = f1_score(true_class, predicted_class_swapped, average='macro')
    f1_micro = f1_score(true_class, predicted_class_swapped, average='micro')
    if output_to_console:
        print("F1 macro: " + str(f1_macro))
        print("F1 micro: " + str(f1_micro))
        
    # Adding matplotlib confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    labels = ['business','politics','sports','health','education','estate','arts','science','technology']
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
    disp.plot()
    plt.title("NYT-Topics")
    plt.xticks(rotation=45)
    plt.show()
        
    if return_tuple:
        return confusion, f1_macro, f1_micro
    else:
        return {
            "confusion": confusion.tolist(),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }
