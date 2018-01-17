#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:42:10 2016

@author: denson
"""

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg',warn = False)
print("backend")
print(matplotlib.get_backend())

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef

import glob


import os



import random

def check_directory(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
        
def get_colorblind_colors():
    # set up plot colors 
    
    # http://tableaufriction.blogspot.ro/2012/11/finally-you-can-use-tableau-data-colors.html
    # https://gist.github.com/AndiH/c957b4d769e628f506bd
    # Tableau Color Blind 10
    tableau20blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
                 (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
                 (255, 188, 121), (207, 207, 207)]   
      
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20blind)):  
        r, g, b = tableau20blind[i]  
        tableau20blind[i] = (r / 255., g / 255., b / 255.)
        
    return tableau20blind
        
    
model = 'ET'
    
sample_size = 1

font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

# set the proper path separator for this operating system
path_sep = os.sep

# get the path to the script  so we can use relative paths
full_path = os.path.realpath(__file__)
script_path, file_name = os.path.split(full_path)

script_version = file_name.split('.')[0]

tableau20blind = get_colorblind_colors()







script_path = script_path + path_sep

# single_model_results_name = 'single_model_results_10percent_samples.csv'

# single_model_results_name = 'single_model_results_1_3_31.csv'

single_model_results_name = 'single_model_results_1_3_15_21_31.csv'


single_model_results_path = script_path + 'keep_results' + path_sep + single_model_results_name

single_model_results_df = pd.read_csv(single_model_results_path, header = 0)

num_win_sizes,cols = np.shape(single_model_results_df)

# read in one bite prediction probas file so we know how much data will be dealing
# with. Then set up dataframes to store the results
window_size = single_model_results_df.ix[0,'window size']
bites_results_input_path = script_path + 'keep_results' + path_sep + \
    'windowed_bites_results_window_' + \
    np.str(window_size) + '_sample_size_' + np.str(sample_size) + 'percent'  \
    + '_' + model + path_sep
    
# get all the results file names for this window size
bites_results_input_filenames = glob.glob(bites_results_input_path + "*.csv")

# read the first results file to get the number of samples and y_true
bites_results_df = pd.read_csv(bites_results_input_filenames[0], header = 0)

y_true = bites_results_df['y_true'].as_matrix()
bite_samples_per_win_size = len(bites_results_input_filenames)

# we have bite samples per window size * the number of window sizes total 
total_running_rows = num_win_sizes * bite_samples_per_win_size




# this will be the columns in our running MCC dataframe for all the bites
# of all the different window sizes
running_headers = ['model',
                   'window size',
                   'sample size %',
                   'IM MCC',
                   'stacked MCC',
                   'min bite MCC',
                   'max bite MCC',
                   'bite #',
                   'running MCC',
                   'bite MCC']
                   
num_running_cols = len(running_headers)
                   
# now we know how much data we have and how much we will create

all_running_array = np.zeros((total_running_rows,num_running_cols))

all_running_df = pd.DataFrame(data = all_running_array, columns = running_headers)

# this is the number samples in the test data
len_y_true = len(y_true)
# make an array for the running sum of the probas for each sample
all_running_proba_sum_array = np.zeros(len_y_true)
# this is the number of prediction probas we have summed so far
# to get the running probas sum
num_all_running_probas = 0

# the all windows all bites row to save
all_running_row = 0    
    
for row in range(num_win_sizes):
    
    window_size = single_model_results_df.ix[row,'window size']

    single_model_MCC = single_model_results_df.ix[row,'MCC']
    
    bites_results_input_path = script_path + 'keep_results' + path_sep + \
        'windowed_bites_results_window_' + \
        np.str(window_size) + '_sample_size_' + np.str(sample_size) + 'percent'  \
        + '_' + model + path_sep
    # get all the results file names for this window size
    bites_results_input_filenames = glob.glob(bites_results_input_path + "*.csv")
    
    # randomize the order of the files
    
    random.shuffle(bites_results_input_filenames)
    
    # we only need about 50
    bites_results_input_filenames = bites_results_input_filenames[0:50]
    
    
    bites_summary_path = script_path + 'keep_results' + path_sep + \
        'windowed_bites_summary_window_'+ np.str(window_size) + '_' + np.str(sample_size) \
        + 'percent_samples_' + model + path_sep
    check_directory(bites_summary_path)
    
    bites_summary_name = 'bites_per_file_performance_window_size_'+ \
        np.str(window_size)  + '_' + np.str(sample_size) \
        + 'percent_samples_' + model + '.csv'
    
    bites_summary_file_path = bites_summary_path + bites_summary_name
    
    # set up a dataframe to store the results for this window size
    
    # we have bite samples per window size rows for the per window running results
    this_running_array = np.zeros((bite_samples_per_win_size,num_running_cols))
    
    this_running_df = pd.DataFrame(data = this_running_array, columns = running_headers)
    
    # make an array for the running sum of the probas for each sample
    this_running_proba_sum_array = np.zeros(len_y_true)
    # this is the number of prediction probas we have summed so far
    # to get the running probas sum
    num_this_running_probas = 0
    
    # read in the prediction probas for each bite of each window size and
    # compute individual bite performance, all stacked performance and stacked
    # performance for just this window size
    
    # this window size bites row to save
    this_running_row = 0
    
    for this_bite_file_path in bites_results_input_filenames:
        

        
        


    
        # read the first results file to get the number of samples and y_true
        bites_results_df = pd.read_csv(this_bite_file_path, header = 0)
    
        bite_rows,bite_cols = np.shape(bites_results_df)
    
        # get the predictions and update the running totals
        this_bite_probas = bites_results_df['proba'].as_matrix()
        
        # this bite performance
        this_bite_class_preds= (this_bite_probas >= 0.5).astype(np.int)
        this_bite_MCC = matthews_corrcoef(y_true,this_bite_class_preds)


        
        # this window size running performance
        this_running_proba_sum_array  = this_running_proba_sum_array  + this_bite_probas
        num_this_running_probas += 1
        this_window_size_running_probas = this_running_proba_sum_array/num_this_running_probas
        this_window_size_running_class_preds = (this_window_size_running_probas >= 0.5).astype(np.int)
        this_window_size_running_MCC = matthews_corrcoef(y_true,this_window_size_running_class_preds)
        

        # all window sizes running MCC
        all_running_proba_sum_array = all_running_proba_sum_array + this_bite_probas
        num_all_running_probas += 1
        all_running_probas = all_running_proba_sum_array/num_all_running_probas
        all_window_size_running_class_preds = (all_running_probas >= 0.5).astype(np.int)
        all_window_size_running_MCC = matthews_corrcoef(y_true,all_window_size_running_class_preds)        
        
        # update the dataframe for this window size
        this_running_df.ix[this_running_row,'window size'] = window_size
        this_running_df.ix[this_running_row,'bite #'] = this_running_row
        this_running_df.ix[this_running_row,'IM MCC'] = single_model_MCC
        this_running_df.ix[this_running_row,'running MCC'] = this_window_size_running_MCC      
        this_running_df.ix[this_running_row,'bite MCC'] = this_bite_MCC 
    
        # update the dataframe for all windows all bites
        all_running_df.ix[all_running_row,'window size'] = window_size
        all_running_df.ix[all_running_row,'bite #'] = all_running_row
        all_running_df.ix[all_running_row,'IM MCC'] = single_model_MCC
        all_running_df.ix[all_running_row,'running MCC'] = all_window_size_running_MCC      
        all_running_df.ix[all_running_row,'bite MCC'] = this_bite_MCC    

        this_running_row += 1
        all_running_row += 1
    

    
    # get the final performance for this window size
    this_window_size_stacked_MCC = this_running_df.ix[(this_running_row - 1),'running MCC']
    this_running_df.ix[:,'stacked MCC'] = this_window_size_stacked_MCC
    

    
    
    
    
    
    
    plot_title = 'Performance with ' + np.str(sample_size) + '% samples - window size ' \
    + np.str(window_size) + '- model ' + model
    
    plot_save_name = 'performance_' +  np.str(sample_size)+ 'percent_samples_window_' \
    + np.str(window_size) +'_model_' + model  + '.png'
    plot_save_path = script_path + 'keep_results' + path_sep + 'plots_and_tables' + path_sep
    check_directory(plot_save_path)
    plot_save_path = plot_save_path + plot_save_name
    
    # if we had a window with less than the full number of bites, discard the extras
    this_running_df = this_running_df[this_running_df['IM MCC'] > 0]


    
    # get the extra stuff we need to plot
    single_model_MCC = this_running_df['IM MCC'].as_matrix()[-1]

    this_running_df['sample size %'] = sample_size
    this_running_df['model'] = model
    this_running_df['window size'] = window_size
    
    stacked_MCC = this_running_df['stacked MCC'].as_matrix()[-1]

    min_MCC = this_running_df['bite MCC'].min()
    
    this_running_df['min bite MCC'] = min_MCC
    
    max_MCC = this_running_df['bite MCC'].max()
    
    this_running_df['max bite MCC'] = max_MCC



    plt.close('all')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
 
    

    
    this_running_df.plot(kind='scatter', 
                         x = 'bite #', 
                         y = 'bite MCC', 
                         ax = ax1, 
                         label = 'bite MCC',
                         color = tableau20blind[3],
                         marker='x')
    this_running_df.plot(kind='scatter',
                         x = 'bite #', 
                         y = 'running MCC', 
                         ax = ax1, 
                         label = 'running MCC',
                         marker='.',
                         color = 'white')
    this_running_df.plot(x = 'bite #', 
                        y = 'stacked MCC', 
                        ax = ax1,
                        linestyle = '-',
                        color = tableau20blind[1])
    this_running_df.plot(x = 'bite #', 
                        y = 'IM MCC', 
                        ax = ax1, 
                        linestyle = '-',
                        linewidth = 1.0,
                        color = tableau20blind[0])
    
    this_running_df.plot(x = 'bite #', 
                        y = 'min bite MCC', 
                        ax = ax1, 
                        linestyle = '--',
                        linewidth = 1.0,
                        color = tableau20blind[2])
    
    this_running_df.plot(x = 'bite #', 
                        y = 'max bite MCC', 
                        ax = ax1, 
                        linestyle = '-.',
                        linewidth = 1.0,
                        color = tableau20blind[2])  
    
    
    max_bite_num = this_running_df['bite #'].max()
    
    if max_bite_num < 99:
        print
    
    
    ax1.set_xlim(0.0,max_bite_num + 5)
    
    
 
    
    ax1.set_ylim(0.0,0.7)
    
    ax1.set_ylabel('MCC')
    
    
    
    ax1.set_title(plot_title)
    
    
    
    
    
    plt.yticks([0.4,
                0.7, 
                single_model_MCC,
                stacked_MCC,
                min_MCC,
                max_MCC])
    
    
    fig.savefig(plot_save_path, dpi = 600, bbox_inches='tight')
    plt.show()
    
    this_running_df = this_running_df[running_headers]
    

    this_running_df.to_csv(bites_summary_file_path, index = False)



    
# plot the all windows all bites dataframe
# get the final performance for all the bites
all_window_size_stacked_MCC = all_running_df.ix[(all_running_row - 1),'running MCC']
all_running_df.ix[:,'stacked MCC'] = all_window_size_stacked_MCC





plot_title = 'Performance with ' + np.str(sample_size) + '% samples - multiple windows'

plot_save_name = 'performance_' + np.str(sample_size) + \
    'percent_samples_multiple windows' + '_model_' + model + '.png'
plot_save_path = script_path + 'keep_results' + path_sep + 'plots_and_tables' + path_sep
check_directory(plot_save_path)
plot_save_path = plot_save_path + plot_save_name

# get the extra stuff we need to plot


stacked_MCC = all_running_df['stacked MCC'].as_matrix()[-1]

min_MCC = all_running_df['bite MCC'].min()

all_running_df['min bite MCC'] = min_MCC

max_MCC = all_running_df['bite MCC'].max()

all_running_df['max bite MCC'] = max_MCC



plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(111)

 



all_running_df.plot(kind='scatter', 
                     x = 'bite #', 
                     y = 'bite MCC', 
                     ax = ax1, 
                     label = 'bite MCC',
                     color = tableau20blind[3],
                     marker='x')
all_running_df.plot(kind='scatter',
                     x = 'bite #', 
                     y = 'running MCC', 
                     ax = ax1, 
                     label = 'running MCC',
                     marker='.',
                     color = 'white')
all_running_df.plot(x = 'bite #', 
                    y = 'stacked MCC', 
                    ax = ax1,
                    linestyle = '-',
                    color = tableau20blind[1])

all_running_df.plot(x = 'bite #', 
                    y = 'min bite MCC', 
                    ax = ax1, 
                    linestyle = '--',
                    linewidth = 1.0,
                    color = tableau20blind[2])

all_running_df.plot(x = 'bite #', 
                    y = 'max bite MCC', 
                    ax = ax1, 
                    linestyle = '-.',
                    linewidth = 1.0,
                    color = tableau20blind[2])  


max_bite_num = all_running_df['bite #'].max()


ax1.set_xlim(0.0,max_bite_num + 5)

ax1.set_ylim(0.0,0.7)

ax1.set_ylabel('MCC')



ax1.set_title(plot_title)





extra_ticks = [stacked_MCC, min_MCC, max_MCC]

plt.yticks([0.4,
            0.7])

plt.yticks(list(plt.yticks()[0]) + extra_ticks)           



fig.savefig(plot_save_path, dpi = 600, bbox_inches='tight')
plt.show()
    
    
    
'''
# redo for patches
rows,cols = np.shape(patches_training_df)

patches_training_df['sample #'] = np.arange(rows)






single_ET_model_array = np.ones(rows)

single_ET_model_array = single_ET_model_array * 0.492466295254145



# Plot the residuals after fitting a linear model , scatter_kws={"s": 80
#sns.lmplot(x = 'sample #', y = 'this file MCC', data =bites_training_df, ci=None,lowess = True)

# sns.rugplot(bites_training_df['this file MCC'])

fig = plt.figure()
ax1 = fig.add_subplot(111)



plot_patches_training_df = patches_training_df.copy(deep = True)
plot_patches_training_df['single model_MCC'] = single_ET_model_array
new_headers = ['stacked MCC', 'sample MCC', 'train file', 'patch #', 'IM MCC']
plot_patches_training_df.columns = new_headers

plot_patches_training_df = plot_patches_training_df.iloc[0:100,:]

plot_patches_training_df.plot(kind='scatter', 
                            x = 'patch #', 
                            y = 'sample MCC', 
                            ax = ax1, 
                            label = 'sample MCC',
                            color = 'darkorange')
plot_patches_training_df.plot(x = 'patch #', y = 'stacked MCC', ax = ax1)
plot_patches_training_df.plot(x = 'patch #', y = 'IM MCC', ax = ax1)

ax1.set_ylim(0.2,0.6)



input_filenames = glob.glob(bites_results_path + "*.csv")

# get the sample IDs
sample_IDs = []

for idx in range(len(input_filenames)):
    this_ID = input_filenames[idx].split('DM4229_with_no_spined_feature_')[1]
    this_ID = this_ID.split('_window_31_bites_predictions.csv')[0]
    this_ID = this_ID.split('_')[1]
    this_ID = np.int(this_ID)
    sample_IDs.append(this_ID)
    
sample_IDs = np.array(sample_IDs)

sort_args = np.argsort(sample_IDs)
sorted_input_filenames = []
for idx in range(len(input_filenames)):
    sorted_input_filenames.append(input_filenames[sort_args[idx]])


# open one file to get the number of rows
proba_results_df = pd.read_csv(sorted_input_filenames[0], header = 0)

headers = proba_results_df.columns.values.tolist()

new_headers = ['probas_sample_0', 'y_pred','y_true']


# save ground truth labels for later
y_true = proba_results_df['y_true'].as_matrix()

proba_results_df.columns =  new_headers

new_headers = ['y_true','probas_sample_0']

proba_results_df = proba_results_df[new_headers]

new_headers = ['probas_sample_0']

for idx in range(1,len(sorted_input_filenames)):
    this_header = 'probas_sample_' + np.str(idx)
    this_proba_df = pd.read_csv(sorted_input_filenames[idx])
    new_headers.append(this_header)
    proba_results_df[this_header] = this_proba_df['proba']

proba_results_array =  proba_results_df.as_matrix()
proba_results_array = proba_results_array[:,1:]

new_headers = ['y_true', 'y_stacked_pred', 'mean_proba', 'std_proba', '2 x std', 'min_proba','max_proba'] + new_headers
proba_results_df['y_true'] = y_true

proba_results_df['mean_proba'] = np.mean(proba_results_array, axis = 1)
proba_results_df['std_proba'] = np.std(proba_results_array, axis = 1)
proba_results_df['2 x std'] = np.std(proba_results_array, axis = 1) * 2
proba_results_df['min_proba'] = np.min(proba_results_array, axis = 1)
proba_results_df['max_proba'] = np.max(proba_results_array, axis = 1)

proba_results_df['y_stacked_pred'] = np.zeros(len(y_true)).astype(np.int)

pred_1_args = np.where(proba_results_df['mean_proba'].as_matrix() >= 0.5)[0]

proba_results_df.ix[pred_1_args,'y_stacked_pred'] = 1

proba_results_df = proba_results_df[new_headers]

proba_results_array_transpose = np.transpose(proba_results_array)

x = np.linspace(0, 15, 31)
data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
ax = sns.tsplot(data=proba_results_array_transpose[:,0:100])

'''
