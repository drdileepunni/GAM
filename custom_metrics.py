
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, classification_report, confusion_matrix, accuracy_score, f1_score, auc
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_auc_score(model, x_test, y_test):
    auc_score = roc_auc_score(y_test, model.predict_proba(x_test)[:,1])
    print("AUC Score: ", auc_score)
    return auc_score

def roc_and_prc(model, X_test, y_test):
    '''
    Plots ROC Curve and PRC Curve

    Parameters:
      model: ML Model object
      X_test: np.array of class test data
      y_test: np.array of target test data

    '''

    # Getting y_pred using predict
    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # Getting y_score using predict_proba
    y_score = model.predict_proba(X_test)[:,1]

    # Print Confusion Matrix
    print('\n\n---Confusion Matrix---')
    print(confusion_matrix(y_test, y_pred))

    # Print Classification Report
    print('\n\n---Classification Report---')
    print(classification_report(y_test, y_pred))

    # ROC Curve
    print('\n\n---ROC Curve Plot---')
    roc_auc = roc_auc_score(y_test, y_score)
    print('\nAUC Score: ', roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC= %0.2f' %roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # PRC Curve
    print('\n\n---PRC Curve Plot---')
    average_precision = average_precision_score(y_test, y_score)
    print('Avg. Precision: ', average_precision)
    precision, recall, thres = precision_recall_curve(y_test, y_score)
    f1, auprc = f1_score(y_test, y_pred), auc(recall, precision)
    print('\nAUC Score: ', auprc)
    plt.plot(recall, precision, marker='.', label='AUC= %0.2f' %auprc)
    plt.title('PRC Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc=0)
    plt.show()
    
def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 
