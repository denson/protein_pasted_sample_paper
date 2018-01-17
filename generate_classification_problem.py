#!/home/denson/anaconda2/bin/python

# -*- coding: utf-8 -*-

print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, \
precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef
from sklearn.cross_validation import StratifiedShuffleSplit



import pandas as pd



path_sep = os.sep

# get the path to the script  so we can use relative paths
full_path = os.path.realpath(__file__)
script_path, file_name = os.path.split(full_path)

script_version = file_name.split('.')[0]

def check_directory(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
output_path = script_path + path_sep + 'synthetic_data' + path_sep

check_directory(output_path)

read_existing = True

#  in each dataset we will have 50/50 class1/class0
sample_size = 200
n_features = 45000
n_informative = 10 
num_noise_feats = n_features - n_informative
mislabel = 0

flip_y = float(mislabel)/100.0


forest_size = 1000
criterion = 'gini'
n_jobs = -1
max_features = None



X_noisy = np.random.rand(sample_size, n_features)

# fairly hard 
X, y = make_classification(n_samples=sample_size , n_features=n_informative, n_redundant=0, n_informative=n_informative,
                           random_state=5, n_clusters_per_class=3, class_sep = 1, flip_y = flip_y, shuffle = True)

# X_sample = np.copy(X)
# y_sample = np.copy(y)

X_noisy[:,0:n_informative] = X[:,:]

train_name = output_path + 'train_noise_feats_' +  \
    np.str(num_noise_feats) + '_mislabeled_' + np.str(mislabel) +   '.csv'
    
test_name = output_path + 'test_noise_feats_' +  \
    np.str(num_noise_feats) + '_mislabeled_' + np.str(mislabel) +   '.csv'
    
    
sss = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=42)
for train_index, test_index in sss:


    
    sorted_train_index = np.sort(train_index)
    sorted_test_index = np.sort(test_index)

    X_train, X_test = X_noisy[train_index,:], X_noisy[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    
    X_sample = np.copy(X_train)
    y_sample = np.copy(y_train)
    
    headers = []
    
    for idx in range(n_informative):
        this_header = 'rel_' + np.str(idx)
        headers.append(this_header)
    
    for idx in range(n_informative,n_features):
        this_header = 'noise_' + np.str(idx)
        headers.append(this_header)    
        
    
        
    train_df = pd.DataFrame(data=X_train, columns = headers)
    train_df['Class'] = y_train
    
    
    test_df = pd.DataFrame(data=X_test, columns = headers)
    test_df['Class'] = y_test
    
    headers = ['Class'] + headers
    
    train_df = train_df[headers]
    train_df.to_csv(train_name, index = False)
    
    
    test_df = test_df[headers]
    test_df.to_csv(test_name, index = False)
    
    '''    
    ic = ExtraTreesClassifier(n_estimators=forest_size, 
            criterion=criterion, n_jobs = n_jobs,\
            max_features=max_features, verbose = 0)  
            
    ic = ic.fit(X_train, y_train)
    
    y_pred = ic.predict(X_test);
    probas = ic.predict_proba(X_test)
    importances = ic.feature_importances_
    

    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    
    print('All features:')
    
    AUC        = roc_auc_score(y_test, probas[:, 1])
    print('AUC = %.4f' % AUC)
    
    MCC        = matthews_corrcoef(y_test, y_pred)
    print('MCC = %.4f' % MCC)
    
    accuracy   = accuracy_score(y_test,y_pred)
    print('accuracy = %.4f' % accuracy)
    
    F1         = f1_score(y_test,y_pred)
    print('F1 = %.4f' % F1)
    
    precision  = precision_score(y_test,y_pred)
    print('precision = %.4f' % precision)
    
    recall     = recall_score(y_test,y_pred)
    print('recall = %.4f' % recall)
    
    
    
    X = X_noisy[:,0:n_informative]
    
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    
    

    ic = ExtraTreesClassifier(n_estimators=forest_size, 
            criterion=criterion, n_jobs = n_jobs,\
            max_features=max_features, verbose = 0)  
            
    ic = ic.fit(X_train, y_train)
    
    y_pred = ic.predict(X_test);
    probas = ic.predict_proba(X_test)
    importances = ic.feature_importances_
    

    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    
    print
    print('Relevant features:')
    
    AUC        = roc_auc_score(y_test, probas[:, 1])
    print('AUC = %.4f' % AUC)
    
    MCC        = matthews_corrcoef(y_test, y_pred)
    print('MCC = %.4f' % MCC)
    
    accuracy   = accuracy_score(y_test,y_pred)
    print('accuracy = %.4f' % accuracy)
    
    F1         = f1_score(y_test,y_pred)
    print('F1 = %.4f' % F1)
    
    precision  = precision_score(y_test,y_pred)
    print('precision = %.4f' % precision)
    
    recall     = recall_score(y_test,y_pred)
    print('recall = %.4f' % recall)
    '''
    
'''    

this_fold = 0
sss = StratifiedShuffleSplit(y_sample, 5, test_size=0.5, random_state=42)
for train_index, test_index in sss:

    X_train_fold, X_test_fold = X_sample[train_index,:],X_sample[test_index,:]
    y_train_fold, y_test_fold = y_sample[train_index], y_sample[test_index]    
    
    ic = ExtraTreesClassifier(n_estimators=forest_size, 
            criterion=criterion, n_jobs = n_jobs,\
            max_features=max_features, verbose = 0)  
            
    ic = ic.fit(X_train_fold, y_train_fold)
    
    y_pred = ic.predict(X_test_fold);
    probas = ic.predict_proba(X_test_fold)
    importances = ic.feature_importances_
    

    fpr, tpr, thresholds = roc_curve(y_test_fold, probas[:, 1])
    print
    print('Fold # %i: ' % this_fold)
    
    AUC        = roc_auc_score(y_test_fold, probas[:, 1])
    print('AUC = %.4f' % AUC)
    
    MCC        = matthews_corrcoef(y_test_fold, y_pred)
    print('MCC = %.4f' % MCC)
    
    accuracy   = accuracy_score(y_test_fold,y_pred)
    print('accuracy = %.4f' % accuracy)
    
    F1         = f1_score(y_test_fold,y_pred)
    print('F1 = %.4f' % F1)
    
    precision  = precision_score(y_test_fold,y_pred)
    print('precision = %.4f' % precision)
    
    recall     = recall_score(y_test_fold,y_pred)
    print('recall = %.4f' % recall)  
    
    this_fold += 1
'''   