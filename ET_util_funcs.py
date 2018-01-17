#!/usr/bin/python


# -*- coding: utf-8 -*-
"""
Created on Sat May 24 20:43:39 2014

@author: densonsmith

Utility functions for scikit learn random forest or extra tree classifiers
"""
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg',warn = True)

import numpy as np
import os
import pylab as pl
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, matthews_corrcoef, auc

import random


def svm_read_problem(txt_file_name, csv_file_name):
    
    """
    
    This is a modification of the standard libSVM function:
    
    svm_read_problem(data_file_name) -> [y, x]

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    
    This will extract just the data values and write to a csv file.
    The row number of each line will be added to the end of each row.
    
    The row numbers are useful for a variety of things.
    
    """
 
     
    output_file      = open(csv_file_name,'w')
    row_number = 0
    for line in open(txt_file_name):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        outline = label + ","
        output_file.write(outline)
        
        for e in features.split():
            ind, val = e.split(":")
            outline = val + ","
            output_file.write(outline)
        outline = np.str(row_number) + "\n"
        output_file.write(outline)
        row_number = row_number + 1
        
        
    output_file.close()
    return 1


def generate_window(win_1_array,window_size,number_of_features,terminal_ID_col):
    
    '''
    win_1_array = input array
    window_size = the size window array we are creating
    number_of_features = number of features in each row of win_1_array, not counting class ID and row #
    terminal_ID_col = column where the terminal ID is, this tells us where sequences start and end.
    
    This function reads in a csv file with each row containing a class ID in
    column #0 and number_of_features additional features in each row. 
    It creates an array with a class ID in column #0 and "windowed" rows.
    
    The value 9999.9999 represents NaN (not a number) in this case.
    
    For example with number_of_features = 3 and window_size = 3 and 12 rows of data:
    
     1,1,2,3
     1,2,3,4
    -1,3,4,5
     1,4,5,6
    -1,5,6,7
    -1,6,7,8
     1,7,8,9
     1,8,9,10
     1,9,10,11
    -1,10,11,12
     1,11,12,13
    -1,12,13,14

    
     windowed data:
    
    
     1,9999.9999,9999.9999,9999.9999,1,2,3,2,3,4
     1,1,2,3,2,3,4,3,4,5
    -1,2,3,4,3,4,5,4,5,6
     1,3,4,5,4,5,6,5,6,7
    -1,4,5,6,5,6,7,6,7,8
    -1,5,6,7,6,7,8,7,8,9
     1,6,7,8,7,8,9,8,9,10
     1,7,8,9,8,9,10,9,10,11
     1,8,9,10,9,10,11,10,11,12
    -1,9,10,11,10,11,12,11,12,13
     1,10,11,12,11,12,13,12,13,14
    -1,11,12,13,12,13,14,9999.9999,9999.9999,9999.9999
    
    
    '''
 
 
    # pull out the class data and the row numbers
    
    row_numbers = win_1_array[:, -1]
    # throw away the row numbers
    win_1_array = win_1_array[:,:-1]
    
    before_and_after_size = (window_size - 1)/2
    m,n = np.shape(win_1_array)
    
    # make a new array that is window_size x number of features wide + 1 more 
    # column for the class information and + 1 more column for the row numbers
    
    window_array = np.ones((m,(number_of_features*window_size)+2))
    
    # 9999.9999 is our representation of nan
    window_array = window_array * 9999.9999
    

    
    row_number = 0
    sequence_count = 0
    
    
    # copy the class values to the window array
    window_array[:,0] = win_1_array[:,0]
    
    sequence_start_row = 0
    sequence_end_row   = 1
    for line in range(m):
        
    
    
            
        if win_1_array[row_number,terminal_ID_col] == 1:
            sequence_count = sequence_count + 1
            sequence_end_row = row_number + 1
            
            thisSequence = win_1_array[sequence_start_row:sequence_end_row,1:]
            
            
            
            # set up to do the middle
            startRow = sequence_start_row 
            endRow   = sequence_end_row 

            startCol = ((before_and_after_size)*number_of_features) + 1
            endCol   = ((before_and_after_size)*number_of_features) + number_of_features + 1
            window_array[startRow:endRow, startCol:endCol] = thisSequence[:,:]
            
            for idx in range(before_and_after_size,0,-1):

                # set up to do the left side
                startRow = sequence_start_row + idx 
                endRow   = sequence_end_row 
                startCol = ((before_and_after_size-idx)*number_of_features) + 1
                endCol   = ((before_and_after_size-idx)*number_of_features) + number_of_features + 1
     
                window_array[startRow:endRow, startCol:endCol] = thisSequence[:-idx,:]

                

                # set up to do the right side
                startRow = sequence_start_row  
                endRow   = sequence_end_row - idx
                startCol = ((before_and_after_size+idx)*number_of_features) + 1
                endCol   = ((before_and_after_size+idx)*number_of_features) + number_of_features + 1

                window_array[startRow:endRow, startCol:endCol] = thisSequence[idx:,:]
                                   
            
            # the next row after the end row is the new start row
            # in python ranges do not include the last index
            # so win_1_array[0:10,:] is the first 10 rows and
            # win_1_array[10:20,:] is the next 10 rows
            sequence_start_row = sequence_end_row
            
    
        row_number = row_number + 1   
        
    window_array[:,-1] = row_numbers
        
    return window_array

def load_dataset(file_name,file_path):
    binary_file_path = file_path + file_name + ".npz"
    try:
        # look for binary file first
        bin_data = np.load(binary_file_path)
        the_data = bin_data['the_data']
    except:
        try:
            csv_file_path = file_path + file_name + ".csv"
            # no binary, look for csv and save binary
            the_data = np.genfromtxt(csv_file_path, dtype=float, delimiter=',')
            # toss class 0 rows, -1 is negative class, 1 is positive class
            the_data = the_data[the_data[:,0] != 0]
            # we might have thrown away rows, we need to redo the row numbers
            numRows,numCols = np.shape(the_data)
            new_row_numbers = range(numRows)
            the_data[:,-1] = new_row_numbers
            np.savez(binary_file_path,the_data)
        except:
            print('failed to load data')
            '''
            # no binary or csv, read libSVM and create both csv and binary
            libSVM_file_path = file_path + file_name + ".txt"
            svm_read_problem(libSVM_file_path,csv_file_path)
            the_data = np.genfromtxt(csv_file_path, dtype=float, delimiter=',')
            # toss class 0 rows, -1 is negative class, 1 is positive class
            the_data = the_data[the_data[:,0] != 0]
            # we might have thrown away rows, we need to redo the row numbers
            numRows,numCols = np.shape(the_data)
            new_row_numbers = range(numRows)
            the_data[:,-1] = new_row_numbers
            np.savez(binary_file_path,the_data)
            '''
            

    return the_data


def compute_performance_metrics(y_test,class_predictions, probas,importances,plot_name, plot_path):        
#def compute_performance_metrics(y_test,class_predictions, probas, importances,plot_name, plot_path):
    
    
    '''
    This works for a binary classifier model only!
    y_test = a vector of "ground truth" classes
    class_predictions = vector of predictions from the model
    probas =  2d array with the probability of the negative class in column #0
              and the probability of the positive class in column #1
              
    importances = an array of feature importances generated by a random forest or
                  extra tree classifier in scikit learn
    
    plot_name = the name of the receiver operating characteristics plot
    
    plot_path = the path for saving the plot, usually it will be "/plots/"
    
    '''
    


    # create the python dict container for results
    resultDict      = {}


    
    
    confusion = confusion_matrix(y_test, class_predictions)
    
    #confusionDict['confusion'] = confusion
    
    
    tp = confusion[1,1].astype(np.float)
    tn = confusion[0,0].astype(np.float)
    fp = confusion[0,1].astype(np.float)
    fn = confusion[1,0].astype(np.float)      
    
    
    print "Statistics:"

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    try:
        roc_auc = roc_auc_score(y_test, probas[:, 1])
    except:
            roc_auc = np.nan
    print "Area under the ROC curve : %f" % roc_auc  
    
    resultDict['roc_auc'] = roc_auc 


    roc_curve_array = np.vstack((fpr,tpr))
    roc_curve_array = np.vstack((roc_curve_array,thresholds))
    roc_curve_array = np.transpose(roc_curve_array)
    

    try:
        accuracy = accuracy_score(y_test, class_predictions )
    except:
        accuracy = np.nan
        
    print "Accuracy score : %f" % accuracy
    
    resultDict['accuracy'] = accuracy

    try:
        precision = precision_score(y_test, class_predictions )
    except:
        precision = np.nan
        
    print "Precision score : %f" % precision  
    
    resultDict['precision'] = precision
    
    try:
        recall = recall_score(y_test, class_predictions )
    except:
        recall = np.nan
        
    print "Recall score : %f" % recall  
    
    resultDict['recall'] = recall

    try:
        sensitivity = tp/(tp + fn)
    except:
        sensitivity = np.nan
        
    print "sensitivity : %f" % sensitivity
    
    resultDict['sensitivity'] = sensitivity
  
    try:
        specificity = tn/(tn + fp)
    except:
        specificity = np.nan
        
    print "specificity : %f" % specificity
    
    resultDict['specificity'] = specificity
    
    try:
        balanced_accuracy = (sensitivity + specificity)/2
    except:
        balanced_accuracy = np.nan
        
    print "balanced accuracy : %f" % balanced_accuracy
        
    resultDict['balanced_accuracy'] = balanced_accuracy
 
    try:
        F1_score = f1_score(y_test, class_predictions )
    except:
        F1_score = np.nan
        
    print "F1 score : %f" % F1_score 
    
    resultDict['F1_score'] = F1_score

    '''
    try:
        num = (tp*tn) - (fp*fn)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        MCC = num/den
    except:
        MCC = np.nan
    '''
    MCC = matthews_corrcoef(y_test, class_predictions)
        
    print "Matthews correlation coefficient : %f" % MCC
    
    resultDict['MCC'] = MCC
    
    # compute optimum threshold
    tpr_minus_fpr = tpr-fpr
                
    indexOptimum = np.argmax(tpr_minus_fpr)
                
    optimumThresholdValue = thresholds[indexOptimum] 
    
    resultDict['optimum_threshold'] = optimumThresholdValue
    
    print "Optimum threshold : %f" % optimumThresholdValue
    
    
    
    # compute Sw (probability excess)
    Sw = sensitivity + specificity - 1
    resultDict['Sw'] = Sw
    
    print "Sw : %f" % Sw


    # compute Sproduct
    Sproduct = sensitivity * specificity 
    resultDict['Sproduct'] = Sproduct
    print "Sproduct : %f" % Sproduct
    
    
    print "Confusion matrix"
    
    
    
    outline = "\t-1\t1"
    print outline
    outline = "-1\t" + np.str(confusion[0,0]) + "\t" + np.str(confusion[0,1])
    print outline
    outline = " 1\t" + np.str(confusion[1,0]) + "\t" + np.str(confusion[1,1])
    print outline
    print
    

    # compute feature importances
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    cumulative_importance = np.cumsum(sorted_importances)

    importances_array = np.vstack((indices,sorted_importances))
    importances_array = np.vstack((importances_array,cumulative_importance))
    importances_array = np.transpose(importances_array)
    
    # we number features starting with 1 but the array starts with 0
    importances_array[:,0] = importances_array[:,0] + 1
    
    

    
    
    # Plot ROC curve
    fig = pl.figure()
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(plot_name)
    pl.legend(loc="lower right")
    
    fig.my_figure_title = plot_name 
    fig.savefig(plot_path + fig.my_figure_title + ".pdf") 
    pl.close(fig)
    
    
    return resultDict,confusion,importances_array,roc_curve_array
    
def extract_window_1_from_window_n(win_1_file,win_n_file,win_size,number_of_features):
    
    '''
    This is a hack to get use out of files that were created before
    I had written the code to create windowed files on the fly.
    
    win_1_file = csv output file
    win_n_file = csv input file
    win_size   = window size of win_n_file
    number_of_features = number of features in window size 1
    
    '''
    windowed_data = np.genfromtxt(win_n_file, dtype=float, delimiter=',')
    class_vector = windowed_data[:,0]
    row_numbers = windowed_data[:,-1]
    
    windowed_features = windowed_data[:,1:-1]
    
    before_size = (win_size-1)/2
    
    start_col = number_of_features * before_size
    end_col   = start_col + number_of_features
    
    new_features = windowed_features[:,start_col:end_col]
    rows,cols = np.shape(windowed_features)
    window_1_array = np.zeros((rows,number_of_features + 2))
    window_1_array[:,0] = class_vector
    window_1_array[:,1:number_of_features+1] = new_features
    window_1_array[:,-1] = row_numbers
    

    fmt = "%d"
    for idx in range(number_of_features):
        fmt = fmt + ",%.6f"
    fmt = fmt + ",%d"

    np.savetxt(win_1_file,window_1_array, fmt=fmt)
    
    return
    
def divide_beg_mid_end(data_array,window_size,number_of_features,terminal_ID_col):

    '''
    This function takes an array of a protein sequence or set of protein sequences
    and breaks it into three parts based on the terminal information in the 
    terminal_ID_col
    
    The function expects column 0 to be 
    the class IDs and the rightmost column to be row numbers.
    
    Inputs:
    data_array = a numpy array of floats
    window_size = the window size for the input data, for example if we are looking
    at 2 amino acids before and after the amino acid to be classified the window_size
    would be 5.  The gnerate_window function has more info on this.
    number_of_features = this is the number of features in the window size 1 dataset 
    from which the windowed data was generated
    terminal_ID_col = this is the column that tells us where protein sequences start
    and stop.
    
    The function expects the termial info to look like this
    
    -1 = beginning of sequence
    -0.8 = 2nd in seq
    -0.6 = 3rd in seq
    -0.4 = 4th in seq
    -0.2 = 5th in seq
     0   = all the amino acids in the middle of the sequence (possible very many)
     0.2 = 5th from the end 
     0.4 = 4th from the end
     0.6 = 3rd from the end
     0.8 = 2nd from the end
     1   = end of sequence
     
     returns:
     begin_seqs = all the amino acids in the first 5 positions, with class and row information
     mid_seqs   = all the amino acids in the middle positions, with class and row information
     end_seqs   = all the amino acids in the last 5 positions, with class and row information
     
     row_num_vec = the last column in the input array, this is useful for reconstructing
     the the predictions in the test set.
     
     '''
    begin_seq       = data_array[data_array[:,terminal_ID_col] < 0]
    mid_seq         = data_array[data_array[:,terminal_ID_col] == 0]
    end_seq         = data_array[data_array[:,terminal_ID_col] > 0]
    row_num_vec     = data_array[:,-1]
    
    return begin_seq,mid_seq,end_seq,row_num_vec
    
    
def generate_sequence_folds(terminal_ID_col,number_of_folds,die_size,input_name,input_path,output_name,output_path):
    '''
    This function will take the input file input_name of protein sequences and divide it into 
    number_of_folds output training and test files. The file names of the folds
    are based on output_name and store in the output_path.
    
    The die_size determines how many sequences are in the test set.  There will be
    about 1/die_size percent in the test set.
    
    number_of_folds = 10
    die_size = 10 # 10 percent in test set
    die_size = 5 # 20 percent in test set
    input_name = "DM_libSVM_Input_WINDOW_SIZE_1"
    input_path    = "/data_sets/"
    output_name = "CV"
    output_path = "/data_sets/"
    
    
    '''
    
    full_path = os.path.realpath(__file__)
    script_path, file = os.path.split(full_path)
    
    csv_data_file_name = script_path + input_path + input_name + ".csv"
    
    input_path = script_path + input_path
    
    train_fold_list = []
    test_fold_list = []
    
    data_array = load_dataset(input_name,input_path)
    for counter in range(0,number_of_folds,1):
        # set up the test and training files for this fold
        print("generating fold # %d" % counter )
        train_file_name = output_name + "_train_fold_" + np.str(counter)
        train_fold_list.append(train_file_name)
        test_file_name  = output_name + "_test_fold_" + np.str(counter)
        test_fold_list.append(test_file_name)
        # make the full file path names
        training_data_file_name = script_path + output_path + train_file_name + ".csv"
        test_file_name = script_path + output_path + test_file_name + ".csv"
        
        training_file = open(training_data_file_name, 'w')
        test_file     = open(test_file_name, 'w')
        
        m,n = np.shape(data_array)
        test_sequences = 0
        # roll to decide if we are writing this sequence to training or test file
        roll = random.randint(1,die_size)
        if roll == 1:
            write_training_flag = False
            test_sequences = test_sequences + 1
        else:
            write_training_flag = True
        row_number = 0
        for line in open(csv_data_file_name):
            
            if write_training_flag:
                training_file.write(line)
                
            else:
                test_file.write(line)
                
            if data_array[row_number,terminal_ID_col] == 1:
                # roll to decide if we are writing the next sequence to training or test file
                # we will put about 10% in the test group
                roll = random.randint(1,die_size)
                if roll == 1:
                    write_training_flag = False
                    test_sequences = test_sequences + 1
                else:
                    write_training_flag = True
            
         
        
            row_number = row_number + 1
                
       
        print "number of test sequences = %d" % test_sequences
        training_file.close()
        test_file.close()    
        
    return train_fold_list, test_fold_list
    
    
def check_directory(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    