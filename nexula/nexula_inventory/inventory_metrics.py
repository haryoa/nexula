from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def nexus_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def nexus_f1_score_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def nexus_f1_score_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')