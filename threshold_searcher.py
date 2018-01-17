#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Created on Sat May 24 20:43:39 2014
    
    @author: densonsmith
    
    This loads the probas, feature importances and y_test values output from
    an ET classifier and searches for a threshold adjustment that will produce a
    specificity as close to 0.85 as possible.
    """
import gc
import csv
#import logging

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg',warn = False)
print("backend")
print(matplotlib.get_backend())

import numpy as np
import pandas as pd
import os



#from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

import pylab as pl


import time


from sklearn.externals import joblib
import cPickle as pickle

from ET_util_funcs import *




# get the path to the script  so we can use relative paths
full_path = os.path.realpath(__file__)
script_path, file = os.path.split(full_path)



# keep up with total time
master_start_time = time.time()

# set up garbage collection debugging
#gc.set_debug(gc.DEBUG_UNCOLLECTABLE|gc.DEBUG_COLLECTABLE)

gc.collect()

#logger = logging.getLogger('ET_test')



#This is the column that contains the indicators of the beginning and end of sequences
# -1 is the beginning and 1 is the end of each sequence
#model_list = ["ET", "RDF"]
#criterion_list = ['entropy','gini']

model = "ET"
criterion = "entropy"

window_size = 31

#target_specificity = 0.85
target_specificity = 0.87

#train_file_name = "DM4080"
#test_file_name = "SL477"

train_file_name = "DM3000"
test_file_name = "DM1229"

plot_name = train_file_name + "_train" + test_file_name + "_test"








input_path    = script_path + "/threshold_search/"


result_path = script_path + "/threshold_search/search_results_" + train_file_name + "_train_" + test_file_name + "_test/"

this_path =  result_path
check_directory(this_path)

plot_path = result_path

csv_file_path = input_path + "unadjusted_probas_" + test_file_name + "_with_spined_feature_window_size_31_ET_entropy.csv"
probas =  np.genfromtxt(csv_file_path, dtype=float, delimiter=',')

csv_file_path = input_path + "unadjusted_predictions_" + test_file_name + "_with_spined_feature_window_size_31_ET_entropy.csv"
class_predictions = np.genfromtxt(csv_file_path, dtype=float, delimiter=',')

csv_file_path = input_path + "y_test_" + test_file_name + "_with_spined_feature_window_size_31_ET_entropy.csv"
y_test = np.genfromtxt(csv_file_path, dtype=float, delimiter=',')

csv_file_path = input_path + "feature_importances_" + train_file_name + "_with_spined_feature_window_size_31_ET_entropy.csv"
importances = np.genfromtxt(csv_file_path, dtype=float, delimiter=',')

indices = importances[:,0].astype(np.int)
indices = indices - 1
importances = importances[:,1]

importances = importances[indices]







win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
    compute_performance_metrics(y_test,class_predictions, probas,importances, plot_name, plot_path)

m,n = np.shape(probas)
adjusted_probas = np.zeros((m,n))
adjusted_class_predictions = class_predictions



this_specificity = win_performance_metrics["specificity"]

threshold_adjustment = 0.00

if win_performance_metrics["specificity"] < target_specificity:
    # we need to increase the specificity
    
    increase_threshold = -0.01
    while(win_performance_metrics["specificity"] < target_specificity):
        previous_threshold_adjustment = threshold_adjustment
        threshold_adjustment = threshold_adjustment + increase_threshold
        probaM, probaN = np.shape(probas)
        for idx in range(probaM):
            adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
            adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
            if adjusted_probas[idx,1] > 1.0:
                adjusted_probas[idx,0] = 0.001
                adjusted_probas[idx,1] = 0.999
            if adjusted_probas[idx,1] >= 0.5:
                adjusted_class_predictions[idx] = 1
            else:
                adjusted_class_predictions[idx] = -1
    
        win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
            compute_performance_metrics(y_test,class_predictions, adjusted_probas,importances, plot_name, plot_path)

        this_specificity = win_performance_metrics["specificity"]
    

        




elif win_performance_metrics["specificity"] > target_specificity:
    # we need to decrease the specificity
    increase_threshold = 0.01
    
    while(win_performance_metrics["specificity"] > target_specificity):
        previous_threshold_adjustment = threshold_adjustment
        threshold_adjustment = threshold_adjustment + increase_threshold
        probaM, probaN = np.shape(probas)
        print("threshold adjustment: %.6f" % threshold_adjustment )
        for idx in range(probaM):
            adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
            adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
            if adjusted_probas[idx,1] > 1.0:
                adjusted_probas[idx,0] = 0.001
                adjusted_probas[idx,1] = 0.999
            if adjusted_probas[idx,1] >= 0.5:
                adjusted_class_predictions[idx] = 1
            else:
                adjusted_class_predictions[idx] = -1
    
        win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
            compute_performance_metrics(y_test,adjusted_class_predictions, adjusted_probas,importances, plot_name, plot_path)


if win_performance_metrics["specificity"] < target_specificity:
    # we need to increase the specificity
    
    increase_threshold = -0.001
    while(win_performance_metrics["specificity"] < target_specificity):
        previous_threshold_adjustment = threshold_adjustment
        threshold_adjustment = threshold_adjustment + increase_threshold
        probaM, probaN = np.shape(probas)
        for idx in range(probaM):
            adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
            adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
            if adjusted_probas[idx,1] > 1.0:
                adjusted_probas[idx,0] = 0.001
                adjusted_probas[idx,1] = 0.999
            if adjusted_probas[idx,1] >= 0.5:
                adjusted_class_predictions[idx] = 1
            else:
                adjusted_class_predictions[idx] = -1
    
        win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
            compute_performance_metrics(y_test,class_predictions, adjusted_probas,importances, plot_name, plot_path)

        this_specificity = win_performance_metrics["specificity"]
    

        




elif win_performance_metrics["specificity"] > target_specificity:
    # we need to decrease the specificity
    increase_threshold = 0.001
    
    while(win_performance_metrics["specificity"] > target_specificity):
        previous_threshold_adjustment = threshold_adjustment
        threshold_adjustment = threshold_adjustment + increase_threshold
        probaM, probaN = np.shape(probas)
        print("threshold adjustment: %.6f" % threshold_adjustment )
        for idx in range(probaM):
            adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
            adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
            if adjusted_probas[idx,1] > 1.0:
                adjusted_probas[idx,0] = 0.001
                adjusted_probas[idx,1] = 0.999
            if adjusted_probas[idx,1] >= 0.5:
                adjusted_class_predictions[idx] = 1
            else:
                adjusted_class_predictions[idx] = -1
    
        win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
            compute_performance_metrics(y_test,adjusted_class_predictions, adjusted_probas,importances, plot_name, plot_path)



if win_performance_metrics["specificity"] < target_specificity:
    # we need to increase the specificity
    
    increase_threshold = -0.0001
    while(win_performance_metrics["specificity"] < target_specificity):
        previous_threshold_adjustment = threshold_adjustment
        threshold_adjustment = threshold_adjustment + increase_threshold
        probaM, probaN = np.shape(probas)
        for idx in range(probaM):
            adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
            adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
            if adjusted_probas[idx,1] > 1.0:
                adjusted_probas[idx,0] = 0.001
                adjusted_probas[idx,1] = 0.999
            if adjusted_probas[idx,1] >= 0.5:
                adjusted_class_predictions[idx] = 1
            else:
                adjusted_class_predictions[idx] = -1
    
        win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
            compute_performance_metrics(y_test,class_predictions, adjusted_probas,importances, plot_name, plot_path)

        this_specificity = win_performance_metrics["specificity"]
    

        




elif win_performance_metrics["specificity"] > target_specificity:
    # we need to decrease the specificity
    increase_threshold = 0.0001
    
    while(win_performance_metrics["specificity"] > target_specificity):
        previous_threshold_adjustment = threshold_adjustment
        threshold_adjustment = threshold_adjustment + increase_threshold
        probaM, probaN = np.shape(probas)
        print("threshold adjustment: %.6f" % threshold_adjustment )
        for idx in range(probaM):
            adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
            adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
            if adjusted_probas[idx,1] > 1.0:
                adjusted_probas[idx,0] = 0.001
                adjusted_probas[idx,1] = 0.999
            if adjusted_probas[idx,1] >= 0.5:
                adjusted_class_predictions[idx] = 1
            else:
                adjusted_class_predictions[idx] = -1
    
        win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
            compute_performance_metrics(y_test,adjusted_class_predictions, adjusted_probas,importances, plot_name, plot_path)



if win_performance_metrics["specificity"] < target_specificity:
    # we need to increase the specificity
    
    increase_threshold = -0.00001
    while(win_performance_metrics["specificity"] < target_specificity):
        previous_threshold_adjustment = threshold_adjustment
        threshold_adjustment = threshold_adjustment + increase_threshold
        probaM, probaN = np.shape(probas)
        for idx in range(probaM):
            adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
            adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
            if adjusted_probas[idx,1] > 1.0:
                adjusted_probas[idx,0] = 0.001
                adjusted_probas[idx,1] = 0.999
            if adjusted_probas[idx,1] >= 0.5:
                adjusted_class_predictions[idx] = 1
            else:
                adjusted_class_predictions[idx] = -1
    
        win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
            compute_performance_metrics(y_test,class_predictions, adjusted_probas,importances, plot_name, plot_path)

        this_specificity = win_performance_metrics["specificity"]
    

        




elif win_performance_metrics["specificity"] > target_specificity:
    # we need to decrease the specificity
    increase_threshold = 0.00001
    
    while(win_performance_metrics["specificity"] > target_specificity):
        previous_threshold_adjustment = threshold_adjustment
        threshold_adjustment = threshold_adjustment + increase_threshold
        probaM, probaN = np.shape(probas)
        print("threshold adjustment: %.6f" % threshold_adjustment )
        for idx in range(probaM):
            adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
            adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
            if adjusted_probas[idx,1] > 1.0:
                adjusted_probas[idx,0] = 0.001
                adjusted_probas[idx,1] = 0.999
            if adjusted_probas[idx,1] >= 0.5:
                adjusted_class_predictions[idx] = 1
            else:
                adjusted_class_predictions[idx] = -1
    
        win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
            compute_performance_metrics(y_test,adjusted_class_predictions, adjusted_probas,importances, plot_name, plot_path)



print("threshold adjustment: %.6f" % threshold_adjustment )
print("previous threshold adjustment: %.6f" % previous_threshold_adjustment )
threshold_adjustment = previous_threshold_adjustment


probaM, probaN = np.shape(probas)
print("threshold adjustment: %.6f" % threshold_adjustment )
for idx in range(probaM):
    adjusted_probas[idx,0] = probas[idx,0] - threshold_adjustment
    adjusted_probas[idx,1] = probas[idx,1] + threshold_adjustment
    if adjusted_probas[idx,1] > 1.0:
        adjusted_probas[idx,0] = 0.001
        adjusted_probas[idx,1] = 0.999
    if adjusted_probas[idx,1] >= 0.5:
        adjusted_class_predictions[idx] = 1
    else:
        adjusted_class_predictions[idx] = -1




# compute performance metrics
plot_name = train_file_name + "_train_" + test_file_name + "_" + np.str(window_size) + "_" + model + "_" + criterion
win_performance_metrics, confusion_matrices, importances_array, roc_curve_array = \
    compute_performance_metrics(y_test,adjusted_class_predictions, adjusted_probas,importances, plot_name, plot_path)
#win_performance_metrics['feature_importances'] = importances_output

win_performance_metrics["threshold_adjustment"] = threshold_adjustment


this_fold_name = test_file_name + "_" + np.str(window_size) + "_" + model + "_" + criterion




csv_name = "performance_metrics_" + test_file_name  + "_window_size_" + np.str(window_size ) + "_" + model + "_" + criterion


csv_path = result_path + csv_name + ".csv"
with open(csv_path,'wb') as f:
    w = csv.DictWriter(f,win_performance_metrics.keys())
    w.writeheader()
    w.writerow(win_performance_metrics)
    f.close()

fmt = "%d, %d"
csv_name = "confusion_matrix_" + test_file_name  + "_window_size_" + np.str(window_size ) + "_" + model + "_" + criterion

csv_path = result_path + csv_name + ".csv"
np.savetxt(csv_path, confusion_matrices, fmt=fmt)

fmt = "%.6f, %.6f"

csv_name = "adjusted_probas_" + test_file_name  + "_window_size_" + np.str(window_size ) + "_" + model + "_" + criterion


csv_path = result_path + csv_name + ".csv"
np.savetxt(csv_path, probas, fmt=fmt)

csv_name = "adjusted_predictions_" + test_file_name  + "_window_size_" + np.str(window_size ) + "_" + model + "_" + criterion
fmt = "%d"
csv_path = result_path + csv_name + ".csv"
np.savetxt(csv_path, class_predictions, fmt=fmt)

csv_name = "feature_importances_" + train_file_name  + "_window_size_" + np.str(window_size ) + "_" + model + "_" + criterion
fmt = "%.6f, %.6f, %.6f"
csv_path = result_path + csv_name + ".csv"
np.savetxt(csv_path, importances_array, fmt=fmt)

csv_name = "roc_curve_" + test_file_name  + "_window_size_" + np.str(window_size ) + "_" + model + "_" + criterion
fmt = "%.6f, %.6f, %.6f"
csv_path = result_path + csv_name + ".csv"
np.savetxt(csv_path, roc_curve_array, fmt=fmt)










