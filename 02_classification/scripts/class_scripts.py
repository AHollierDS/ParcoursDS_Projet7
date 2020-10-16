"""
This file contains useful scripts for classification purposes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_auc_score, roc_curve

def plot_roc(y_true, y_proba):
    """
    Plots the False Positive Rate against True Positive Rate curve for a given prediction.
    Also gives the prediction score (AUC) and optimal 
    """
    # ROC curve characteristics
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    score = round(roc_auc_score(y_true, y_proba),3)
    
    # Optimum
    dist = fpr**2 + (1-tpr)**2
    i_opt = list(dist).index(dist.min())
    thr_opt = 100*round(thr[i_opt],3)
    
    # Plotting
    plt.plot(fpr, tpr, label = 'ROC curve')
    plt.scatter(fpr[i_opt], tpr[i_opt], marker = 'o', color = 'r', label = f'Optimum : thr = {thr_opt}%')
    plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'lightgrey')
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc = 'lower right')
    plt.title(f'ROC curve - score = {score}')
    