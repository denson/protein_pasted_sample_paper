#!/home/denson/anaconda2/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:09:59 2016

@author: densonsmith
"""

#%%

# import some tools that are specific to the operating system we are 
import os

import glob

import numpy as np
import pandas as pd


from ET_util_funcs import *

from sklearn.cross_validation import StratifiedShuffleSplit

from random import shuffle



# This function checks to see if a directory exists
# if it does not exists it creates it

def check_directory(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


print

# set the proper path separator for this operating system
path_sep = os.sep
# get the path to the script  so we can use relative paths
    
# this tells the us where the script is located on the disk
full_path = os.path.realpath(__file__)
script_path, file_path = os.path.split(full_path)

# this adds the correct path seperator for this OS at the end of the path
script_path = script_path + path_sep

# this gets the name of the script that is running. Often it is useful to save
# this in our output so we know which version created the output
script_version = file_path.split('.')[0]



#%%    


number_of_features = 56
terminal_ID_col = 56

window_size = 3

number_of_bite_sample_files = 100

# This is the sample size for each bite file. It is converted to sample_size/100
# for input into the StratifiedShuffleSplit utility
sample_size = 1

float_sample_size = np.float(sample_size)

sample_fraction = float_sample_size  / 100.0


# This allows us to selectively create only some of the datasets

# Create new individual protein files?
create_new_windowed_individual_protein_files = False

# Create bites?
create_new_bites = True

# Create patches?
create_new_patches = False

# Create new windowed test data?
create_new_windowed_test_set = True

# this is how many patches we take from each bite sample
# the total number of patches will be:
# number_of_bite_sample_files * number_of_patch_samples_per_file
number_of_patch_samples_per_file = 100

individual_proteins_output_path = script_path + 'individual_protein_files' + path_sep

check_directory(individual_proteins_output_path)

windowed_individual_protein_path = script_path + 'windowed_individual_protein_files_window_' + np.str(window_size) + path_sep

check_directory(windowed_individual_protein_path)

windowed_bites_output_path = script_path + 'windowed_bites_files_window_' + \
    np.str(window_size) + '_sample_size_' + np.str(sample_size) + 'percent' + path_sep

check_directory(windowed_bites_output_path)

windowed_patches_output_path = script_path + 'windowed_patches_files_window_' \
    + np.str(window_size) + '_sample_size_' + np.str(sample_size) + 'percent' + path_sep

check_directory(windowed_patches_output_path)

input_path    = script_path + "data_sets" + path_sep
#train_file_name = "DM4080_with_spined_feature"
#train_file_name = "DM3000_with_spined_feature"
# train_file_name = "DM4229_with_no_spined_feature.csv"
# test_file_name  = "SL329_with_no_spined_feature.csv"

test_file_name = "DM4229_with_no_spined_feature.csv"
train_file_name  = "SL329_with_no_spined_feature.csv"

train_dataset_id = train_file_name.split('.')[0]

train_file_path = input_path + train_file_name





#test_file_name_list         = ["SL477_with_spined_feature"]
#test_file_name_list         = ["DM1229_with_spined_feature"]


test_file_path = input_path + test_file_name

# read in training data

input_array = np.genfromtxt(train_file_path, dtype=float, delimiter=',')

rows,cols = np.shape(input_array)


print('there are %i training rows' % rows)
class_row_array = np.zeros((rows,2))

class_row_array[:,0] = input_array[:,0]
class_row_array[:,1] = input_array[:,-1]


class_row_headers = ['class','row_number']
class_row_df = pd.DataFrame(data = class_row_array, columns = class_row_headers)

class_row_df.to_csv('class_row_data.csv', index = False)

row_number = 0
sequence_count = 0

sequence_start_row = 0
sequence_end_row   = 1

# make headers
headers = ['class']
total_features = number_of_features * window_size

for idx in range(total_features):
    this_feature_name = 'f_' + np.str(idx)
    headers.append(this_feature_name)
    
headers.append('row_number')


#%%

if not create_new_windowed_individual_protein_files:
    print('using existing windowed individual protein files')
else:
        
    print('creating windowed individual protein files')
    
    
    for line in range(rows):
        if input_array[row_number,terminal_ID_col] == 1:
            sequence_count = sequence_count + 1
            sequence_end_row = row_number + 1
            
            this_protein_array = input_array[sequence_start_row:sequence_end_row]
            
            this_protein_file_name = train_dataset_id + '_protein_' + np.str(sequence_count) + '.csv'
            this_output_file_path = individual_proteins_output_path + this_protein_file_name
            
            np.savetxt(this_output_file_path, this_protein_array, fmt='%.6f', delimiter=',')
            print('saved protein %i file' % sequence_count)
            
            sequence_start_row = sequence_end_row
            
            # make the windowed data array
            this_protein_windowed_array = generate_window(this_protein_array,window_size,number_of_features,terminal_ID_col)
            this_protein_file_name = train_dataset_id + '_protein_' + np.str(sequence_count) + '_windowed_' + np.str(window_size) + '_bite.csv'
            this_output_file_path = windowed_individual_protein_path + this_protein_file_name
            
            output_df = pd.DataFrame(data = this_protein_windowed_array, columns = headers)
            # np.savetxt(this_output_file_path,this_protein_windowed_array,  fmt='%.6f', delimiter=',')
            output_df.to_csv(this_output_file_path,index = False)
            
            
    
        row_number = row_number + 1 
    
#%%
    
# get all the filenames we just made
input_filenames = glob.glob(windowed_individual_protein_path + "*.csv")
    
y_train = class_row_df['class'].as_matrix()
sss_rows = class_row_df['row_number'].as_matrix()
    
sss = StratifiedShuffleSplit(y_train, n_iter = number_of_bite_sample_files, test_size=sample_fraction, random_state=0)

# convert the StratifiedShuffleSplit to python sets for efficient searching
sss_row_sets = []
for train_index,test_index in sss:
    sss_row_sets.append(set(sss_rows[test_index]))

if not create_new_bites:
    print('keeping existing bites files')
    
else:
    
    # create the bite sample output files
    sample_output_file_names = []
    for idx in range(number_of_bite_sample_files):
        this_file_name = windowed_bites_output_path + train_dataset_id + '_sample_' + np.str(idx) + '_window_' + np.str(window_size) + '_bites.csv'
        sample_output_file_names.append(this_file_name)
    sample_f = [open(sample_output_file_names[i], "w") for i in range(number_of_bite_sample_files)]
    
    
    # write the headers to all the sample files
    string_header = ','.join(headers) + '\n'
    for idx in range(number_of_bite_sample_files):
        sample_f[idx].write(string_header)
    
    for this_file in input_filenames:
        this_file_name = this_file.split(path_sep)[-1]
        print('processing file %s' % this_file_name)
        with open(this_file) as input_f:
            # skip the header
            first_line = input_f.readline()
            for line in input_f:
                # strip the newline character from the end, save the result in a new
                # string to preserve the original
                this_line = line.strip()
                # convert line string into a python list
                features = this_line.split(',')
                # the sample number is at the very end
                this_row_number = float(features[-1])
                # loop through all the output files and decide whether to write
                # this sample to each
                for idx in range(number_of_bite_sample_files):
                    # the StratifiedShuffleSplit has tuples of train_index,test_index
                    # we are only interested in the test_index
    
                    # if the current row is in this row set we want to 
                    # include it
    
                    if float(this_row_number) in sss_row_sets[idx]:
                        sample_f[idx].write(line)
    
    
    # close all the output files
    for fh in sample_f:
        fh.close()
    
#%%

if not create_new_patches:
    print('keeping existing patches files')
    
else:
        
    # create the patches sample files
    # we start with the bites samples
    print('creating patches')
    input_filenames = glob.glob(windowed_bites_output_path + "*.csv")
    
    # loop through all the files and create some patches sample
    for this_file in input_filenames:
        this_file_name = this_file.split(path_sep)[-1]
        print('processing file %s to patches' % this_file_name)
        this_file_id = this_file_name.split(path_sep)[0]
        this_file_id = this_file_id.split('.')[0]
        # get the parts of the headers
        class_header = headers[0]
        row_number_header = headers[-1]
        feature_headers = headers[1:-1]
        bite_df = pd.read_csv(this_file, header = 0)
        
        # this is how many features we will include in each patch
        # if we have n features, sqrt(n) is a common value for RDF and ET
        max_features = np.floor(np.sqrt(len(feature_headers))).astype(int)
        
        # now we will repeatedly sample the features from this bite sample
        for idx in range(number_of_patch_samples_per_file):
            # shuffle the feature_headers 
            shuffle(feature_headers)
            
            # if we take the first max_features features then we have a random
            # sample of the features
            this_feature_sample = feature_headers[0:max_features]
            # add the class and sample row number headers
            this_feature_sample.insert(0,class_header)
            this_feature_sample.insert(len(this_feature_sample), row_number_header)
            this_patch_df = bite_df[this_feature_sample]
            this_patch_output_name = this_file_id + '_patch_' + np.str(idx) + '.csv'
            this_patch_output_path = windowed_patches_output_path + this_patch_output_name
            this_patch_df.to_csv(this_patch_output_path, index = False)
        
        
        
        

    



#%%
if not create_new_windowed_test_set:
    print('keeping existing windowed test set')
    
else:
    # create and save a windowed test file
    print('creating windowed test set')
    input_array = np.genfromtxt(test_file_path, dtype=float, delimiter=',')
    # make the windowed data array
    this_protein_windowed_array = generate_window(input_array,window_size,number_of_features,terminal_ID_col)
    this_dataset_name = test_file_name.split('.')[0]
    this_protein_file_name = this_dataset_name + '_windowed_size_' + np.str(window_size) + '.csv'
    this_output_file_path = input_path + this_protein_file_name
    
    output_df = pd.DataFrame(data = this_protein_windowed_array, columns = headers)
    # np.savetxt(this_output_file_path,this_protein_windowed_array,  fmt='%.6f', delimiter=',')
    output_df.to_csv(this_output_file_path,index = False)
        