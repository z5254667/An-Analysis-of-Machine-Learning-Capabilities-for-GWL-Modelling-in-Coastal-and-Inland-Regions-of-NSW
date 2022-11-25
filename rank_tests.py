#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 09:28:25 2022

@author: johnsalvaris
"""

# Import packages
from sklearn.linear_model import LinearRegression
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

bore_id = 'GW075025.1.1'  # 'GW080415.1.1', 'GW075405.1.1', 'GW075025.1.1', 'GW403746.3.3', 'GW081101.1.2'
first_test = 320
last_test = 345


if first_test > last_test:
    raise Exception("First test is greater than last test")

for i in range(first_test, last_test + 1):
    path = Path(os.getcwd()) / f'{bore_id}/multi_tests/test_{i}/test_{i}_all_scores.xlsx'
    lstm_test_df = pd.read_excel(path, sheet_name='lstm_test')
    lstm_train_df = pd.read_excel(path, sheet_name='lstm_train')
    svr_test_df = pd.read_excel(path, sheet_name='svr_test')
    svr_train_df = pd.read_excel(path, sheet_name='svr_train')

    if i == first_test:  # Create Dictionary
        keys = lstm_test_df.columns[1:]
        keys = list(map(lambda x: f'Average {x}', keys))
        keys.insert(0, 'Test')
        keys = np.asarray(keys)
        lstm_test_values = [[] for i in keys]
        lstm_train_values = [[] for i in keys]
        svr_test_values = [[] for i in keys]
        svr_train_values = [[] for i in keys]

        tested_input_combs = []  # List to store the input combinations tested
        tested_dataset_lengths = []  # List to store the dataset length tested
        tested_lags = []  # List to store the input lags tested
        tested_average_period = []  # List to store period size tested
        tested_output_combs = []  # List to store the output combinations tested
        tested_train_samples = []  # List to store the number of training samples tested
        tested_test_samples = []  # List to store the number of testing samples tested
        tested_total_samples = []  # List to store the number of combined training and testing samples tested

        dataset_length_missing = False
        samples_missing = False

    # Test Number
    lstm_test_values[0].append(i)
    lstm_train_values[0].append(i)
    svr_test_values[0].append(i)
    svr_train_values[0].append(i)

    # Average scores
    for j in list(lstm_test_df.columns[1:]):
        index = np.where(np.array(lstm_test_df.columns[1:]) == j)[0][0] + 1
        lstm_test_values[index].append(np.average(lstm_test_df[j]))
        lstm_train_values[index].append(np.average(lstm_train_df[j]))
        svr_test_values[index].append(np.average(svr_test_df[j]))
        svr_train_values[index].append(np.average(svr_train_df[j]))

    # Add to list of input combinations
    for file in os.listdir(Path(os.getcwd()) / f'{bore_id}/multi_tests/test_{i}'):
        if file.endswith(".xlsx") and file[0] == 'r':
            sample_run_df = pd.read_excel(Path(os.getcwd()) / f'{bore_id}/multi_tests/test_{i}' / file)
            break

    param = sample_run_df['Unnamed: 0'].to_numpy()
    param_val = sample_run_df['Parameters'].to_numpy()
    tested_input_combs.append(list(param_val[list(map(lambda x: x[:5] == 'Input', param))]))
    tested_lags.append(param_val[param == 'Lag(s)'][0])
    tested_average_period.append(param_val[param == 'Average Period'][0])
    tested_output_combs.append(param_val[param == 'Output'][0])
    try:
        tested_dataset_lengths.append(param_val[param == 'Years of Data Provided'][0])
    except IndexError:
        if not dataset_length_missing:
            print('Tested Dataset Lengths Unavailable')
            dataset_length_missing = True
    try:
        tested_train_samples.append(param_val[param == 'Train Set Samples'][0])
        tested_test_samples.append(param_val[param == 'Test Set Samples'][0])
        tested_total_samples.append(param_val[param == 'Train Set Samples'][0] + param_val[param == 'Test Set Samples'][0])
    except IndexError:
        if not samples_missing:
            print('Tested Train/Test Set Samples Unavailable')
            samples_missing = True

# Dataframes with the ranks
lstm_test_ranks = pd.DataFrame({keys[i]: lstm_test_values[i] for i in range(len(keys))})
lstm_train_ranks = pd.DataFrame({keys[i]: lstm_train_values[i] for i in range(len(keys))})
svr_train_ranks = pd.DataFrame({keys[i]: svr_train_values[i] for i in range(len(keys))})
svr_test_ranks = pd.DataFrame({keys[i]: svr_test_values[i] for i in range(len(keys))})


# Function to print ranked tests in the console
def test_ranks(df, score):
    if score in ['Average Coefficient of Determination', 'Average Normalised Nash Sutcliffe Efficiency', "Average Pearson's Correlation Coefficient", 'Average Index of Agreement',
                 'Average Kling-Gupta Efficiency']:
        ranked_df = df.sort_values(by=[score], ascending=False)
    else:
        ranked_df = df.sort_values(by=[score])
    show_df = pd.concat((ranked_df['Test'], ranked_df[score]), axis=1)
    ranked_scores = ranked_df[score].to_numpy()
    ranked_inputs = np.array([])
    ranked_lengths = np.array([])
    for i in show_df.index:
        ranked_inputs = np.append(ranked_inputs, tested_input_combs[i])  # Note using global variable tested_input_combs
        if not dataset_length_missing:
            ranked_lengths = np.append(ranked_lengths, tested_dataset_lengths[i])  # Note using global variable tested_dataset_lengths
    return show_df, ranked_scores, ranked_inputs, ranked_lengths


svr_r2_df, svr_r2, svr_r2_input, svr_r2_length = test_ranks(svr_test_ranks, 'Average Coefficient of Determination')
svr_rmse_df, svr_rmse, svr_rmse_input, svr_rmse_length = test_ranks(svr_test_ranks, 'Average Root Mean Squared Error')
lstm_r2_df, lstm_r2, lstm_r2_input, lstm_r2_length = test_ranks(lstm_test_ranks, 'Average Coefficient of Determination')
lstm_rmse_df, lstm_rmse, lstm_rmse_input, lstm_rmse_length = test_ranks(lstm_test_ranks, 'Average Root Mean Squared Error')

length_dictionary = {'lstm_r2': [lstm_r2_length, lstm_r2, lstm_r2_df],
                     'lstm_rmse': [lstm_rmse_length, lstm_rmse, lstm_rmse_df],
                     'svr_r2': [svr_r2_length, svr_r2, svr_r2_df],
                     'svr_rmse': [svr_rmse_length, svr_rmse, svr_rmse_df]}


####
# # Quick results
print('')
print('SVR r^2:')
print(svr_r2_df.sort_index())  # remove .sort_index to put in RNAK ORDER, not modelled order
print('')
print('SVR RMSE:')
print(svr_rmse_df.sort_index())  # remove .sort_index to put in RNAK ORDER, not modelled order
print('')
print('LSTM r^2:')
print(lstm_r2_df.sort_index())  # remove .sort_index to put in RNAK ORDER, not modelled order
print('')
print('LSTM RMSE:')
print(lstm_rmse_df.sort_index())  # remove .sort_index to put in RNAK ORDER, not modelled order
# print(tested_input_combs)
print(tested_dataset_lengths)

