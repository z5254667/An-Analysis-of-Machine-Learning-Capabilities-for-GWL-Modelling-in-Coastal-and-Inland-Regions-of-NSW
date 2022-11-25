#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:22:57 2022

@author: johnsalvaris
"""

# Import created scripts
import import_functions

from main_script import main_script

# Import packages
import itertools
import os
import PyPDF2
import shutil
import sys
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PyPDF2.errors import EmptyFileError


bore_id = 'GW080415.1.1'  # 'GW036872.1.1', 'GW075025.1.1', 'GW075405.1.1', 'GW080079.1.1' 'GW080415.1.1', 'GW080980.1.1', 'GW081101.1.2', 'GW273314.1.1', 'GW403746.3.3'

gwl_output = 'average_delta'  # 'standard', 'delta', 'average', 'average_delta'

gwl_input = 'average_delta'  # 'standard', 'delta', 'none', 'average', 'average_delta'
other_variables = ['av_daily_rain', 'av_max_temp', 'av_vp', 'av_evap_syn', 'av_radiation', 'av_rh_tmin', 'av_et_morton_actual', 'av_mslp']


years_of_data = 'all'  # 'all' or number

av_period = 30  # days to average
periods_in = 3  # time lags

# other_variables = list(map(lambda x: list(x), itertools.combinations(other_variables, 2)))
# other_variables = list(map(lambda x: list(x), itertools.combinations(other_variables, len(other_variables) - 2)))


for ov in other_variables:
# for yd in years_of_data:
    # other_variables_copy = other_variables.copy()
    # other_variables_copy.remove(ov)

    # Initialise runs in test
    test_runs = 10
    run = 1
    times_list = []  # Stores unique time identifier of each run

    while run <= test_runs:  # Choose number of runs
        try:
            print('')
            print(f'Run {run}')
            print('')
            current_run = main_script(use_awo=False)
            current_run.run(bore_id, gwl_output, gwl_input,
                            [ov],
                            years_of_data, av_period, periods_in)
        except TimeoutError:  # Restart run if times out
            print('')
            print(f'TimeoutError - Retry Run {run}: {dt.datetime.now()}')
            print('')
        except EmptyFileError:  # Restart run if empty file
            print('')
            print(f'EmptyFileError - Retry Run {run}: {dt.datetime.now()}')
            print('')
        except KeyboardInterrupt:  # Breaks loop manually
            print('')
            print(f'KeyboardInterrupt - Exit code at: {dt.datetime.now()}')
            print('')
            sys.exit()
        # except: # Restart run if other error
        #     print('')
        #     print(f'Error - Retry Run {run}: {dt.datetime.now()}')
        #     print('')
        else:
            times_list.append(current_run.start_date_time.strftime("%Y_%m_%d__%H_%M_%S"))
            run += 1

    # Path to overall file
    bore_path = Path(os.getcwd()) / f'{current_run.bore_id}'

    # Create working folder number
    n = 1
    folder_needed = True
    while folder_needed:
        try:
            os.mkdir(bore_path / f'multi_tests/test_{n}')
        except FileExistsError:
            n += 1
        else:
            test_path = bore_path / f'multi_tests/test_{n}'
            folder_needed = False

    # Initialise dictionaries for scores
    lstm_train_dict = {}
    lstm_test_dict = {}
    svr_train_dict = {}
    svr_test_dict = {}

    for i in range(len(times_list)):
        t = times_list[i]
        # Copy outputs of each run to the test folder
        shutil.copyfile(bore_path / f'logs/output_log_{t}.pdf', test_path / f'run_{i + 1}_test_{n}_output_log_{t}.pdf')
        shutil.copyfile(bore_path / f'spreadsheets/output_spreadsheet_{t}.xlsx', test_path / f'run_{i + 1}_test_{n}_output_spreadsheet_{t}.xlsx')
        shutil.copyfile(bore_path / f'models/svr_model_{t}.pkl', test_path / f'run_{i + 1}_test_{n}_svr_model_{t}.pkl')
        shutil.copytree(bore_path / f'models/lstm_model_{t}', test_path / f'run_{i + 1}_test_{n}_lstm_model_{t}')

        # Add scores from the run to the dictionary
        lstm_scores = pd.read_excel(test_path / f'run_{i + 1}_test_{n}_output_spreadsheet_{t}.xlsx', sheet_name='lstm_scores')
        svr_scores = pd.read_excel(test_path / f'run_{i + 1}_test_{n}_output_spreadsheet_{t}.xlsx', sheet_name='svr_scores')

        if len(lstm_train_dict) == 0:
            for i in range(len(lstm_scores['score'])):
                lstm_train_dict[lstm_scores['score'][i]] = []
                lstm_test_dict[lstm_scores['score'][i]] = []
                svr_train_dict[svr_scores['score'][i]] = []
                svr_test_dict[svr_scores['score'][i]] = []
        for i in range(len(lstm_scores['score'])):
            lstm_train_dict[lstm_scores['score'][i]].append(lstm_scores['train'][i])
            lstm_test_dict[lstm_scores['score'][i]].append(lstm_scores['test'][i])
            svr_train_dict[svr_scores['score'][i]].append(svr_scores['train'][i])
            svr_test_dict[svr_scores['score'][i]].append(svr_scores['test'][i])

    # Create temporary pdf to show average scores
    summary_scores_text_path = import_functions.make_path(f'{current_run.bore_id}/temp', 'score_summary_text.pdf')
    summary_scores_text = FPDF()
    summary_scores_text.add_page()
    summary_scores_text.set_font('Courier', size=8)

    # Create temporary pdf to show general test information
    summary_general_text_path = import_functions.make_path(f'{current_run.bore_id}/temp', 'summary_general_text.pdf')
    summary_general_text = FPDF()
    summary_general_text.add_page()
    summary_general_text.set_font('Courier', size=8)

    # Create general pdf to show boxplots
    summary_scores_graph_path = import_functions.make_path(f'{current_run.bore_id}/temp', 'score_summary_graph.pdf')
    summary_scores_graph = PdfPages(summary_scores_graph_path)

    # Create LSTM Train boxplot
    plt.clf()
    lstm_train = pd.DataFrame(lstm_train_dict)
    lstm_train.boxplot(figsize=(8.27, 11.69), showmeans=True, rot=90)
    plt.title('LSTM Train Scores')
    plt.tight_layout()
    plt.savefig(summary_scores_graph, format='pdf')
    plt.close()

    # Create LSTM Test boxplot
    plt.clf()
    lstm_test = pd.DataFrame(lstm_test_dict)
    lstm_test.boxplot(figsize=(8.27, 11.69), showmeans=True, rot=90)
    plt.title('LSTM Test Scores')
    plt.tight_layout()
    plt.savefig(summary_scores_graph, format='pdf')
    plt.close()

    # Create SVR Train boxplot
    plt.clf()
    svr_train = pd.DataFrame(svr_train_dict)
    svr_train.boxplot(figsize=(8.27, 11.69), showmeans=True, rot=90)
    plt.title('SVR Train Scores')
    plt.tight_layout()
    plt.savefig(summary_scores_graph, format='pdf')
    plt.close()

    # Create SVR Test boxplot
    plt.clf()
    svr_test = pd.DataFrame(svr_test_dict)
    svr_test.boxplot(figsize=(8.27, 11.69), showmeans=True, rot=90)
    plt.title('SVR Test Scores')
    plt.tight_layout()
    plt.savefig(summary_scores_graph, format='pdf')
    plt.close()

    summary_scores_graph.close()

    # Text for scores summary pdf
    lines = []
    lines.append('<><> LSTM <><>')
    lines.append('')

    for i in lstm_train:
        char_len = len(i)  # Max character length = 36 for alignment
        lines.append(f'Average Train {i}:{" " * (36 - char_len)} \t {np.mean(lstm_train[i])}')  # Calculates average score
    lines.append('')

    for i in lstm_test:
        char_len = len(i)  # Max character length = 36 for alignment
        lines.append(f'Average Test {i}:{" " * (36 - char_len)} \t {np.mean(lstm_test[i])}')  # Calculates average score
    lines.append('')

    lines.append('<><> SVR <><>')
    lines.append('')

    for i in svr_train:
        char_len = len(i)  # Max character length = 36 for alignment
        lines.append(f'Average Train {i}:{" " * (36 - char_len)} \t {np.mean(svr_train[i])}')  # Calculates average score
    lines.append('')

    for i in svr_test:
        char_len = len(i)  # Max character length = 36 for alignment
        lines.append(f'Average Test {i}:{" " * (36 - char_len)} \t {np.mean(svr_test[i])}')  # Calculates average score

    for line in lines:
        summary_scores_text.cell(0, 5, txt=line, ln=1, align='L')
    summary_scores_text.output(summary_scores_text_path)

    # Text for general info pdf
    gen_df = pd.read_excel(test_path / f'run_{test_runs}_test_{n}_output_spreadsheet_{t}.xlsx')
    lines = []
    lines.append('<><> General Information <><>')
    lines.append('')
    lines.append(f'Test: {n}')
    for i in range(len(gen_df)):
        lines.append(f"{gen_df['Unnamed: 0'][i]}: {gen_df['Parameters'][i]}")
    lines.append('')
    lines.append('<><> Run Identifiers <><>')
    for i in range(len(times_list)):
        lines.append(f'Run {i + 1}: {times_list[i]}')

    for line in lines:
        summary_general_text.cell(0, 5, txt=line, ln=1, align='L')
    summary_general_text.output(summary_general_text_path)

    # Include Input Correlation Heatmap
    temp_input_graphs_path_4 = import_functions.make_path(f'{current_run.bore_id}/temp', 'temp_input_graphs_4.pdf')

    # Combine into single pdf
    summary_file = test_path / f'test_{n}_summary.pdf'

    pdfs = [summary_general_text_path, summary_scores_text_path, summary_scores_graph_path, temp_input_graphs_path_4]

    merger = PyPDF2.PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(summary_file)
    merger.close()

    with pd.ExcelWriter(test_path / f'test_{n}_all_scores.xlsx') as writer:
        lstm_train.to_excel(writer, sheet_name='lstm_train')
        lstm_test.to_excel(writer, sheet_name='lstm_test')
        svr_train.to_excel(writer, sheet_name='svr_train')
        svr_test.to_excel(writer, sheet_name='svr_test')

    print('')
    print(f'Test {n} complete: {dt.datetime.now()}')
    print('')
