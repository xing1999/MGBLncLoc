from sklearn.metrics import precision_score, recall_score, f1_score,precision_recall_fscore_support, accuracy_score,roc_auc_score
import numpy as np

def convert_to_labels(y_hat):
    n, m = y_hat.shape
    labels=[0]*n

    for i in range(n):
        for j in range(m):
            if y_hat[i, j] == 1:
                labels[i]=j
    return labels
def per_eva(list_label,list_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(list_label, list_pred, average=None, labels=[0, 1, 2, 3])
    accuracy_per_class = {}
    for i in set(list_label):
        indices = [j for j, x in enumerate(list_label) if x == i]
        accuracy_per_class[i] = accuracy_score(np.array(list_label)[indices], np.array(list_pred)[indices])
    for i in range(4):
        print("label-{}:  Precision:{}  Recall:{}  F1-Score:{}  Accuracy_per:{} ".format(i, precision[i], recall[i],
                                                                                         f1[i], accuracy_per_class[i]))
    return precision,recall,f1,accuracy_per_class


def evaluate(y_hat, y):
    list_pred = convert_to_labels(y_hat)
    list_label = convert_to_labels(y)
    # print(list_label)
    # print("pred:{}".format(list_pred))
    macro_precision = precision_score(list_label, list_pred, average='macro')
    macro_recall = recall_score(list_label, list_pred, average='macro')
    macro_F1_Score = f1_score(list_label, list_pred, average='macro')
    macro_accuracy = accuracy_score(list_label, list_pred)
    average_auc = roc_auc_score(y, y_hat, average='macro')
    return list_label,list_pred,macro_precision, macro_recall, macro_F1_Score, macro_accuracy,average_auc
