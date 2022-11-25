#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:42:59 2022

@author: johnsalvaris
"""

# Import created scripts
import math
import matplotlib.pyplot as plt
import seaborn as sns
import import_functions
import bore_class
import model_parameters_class
import keras_lstm_class
import score_functions
import sklearn_svr_class
import output_summary_class

# Import packages
import datetime as dt
import numpy as np
import pandas as pd

from time import time
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE


class main_script():
    def __init__(self, use_awo=True):
        self.use_awo = use_awo

    def run(self, bore_id, gwl_output='standard', gwl_input='standard', other_variables=None, years_of_data='all', av_period=30, periods_in=3):
        print(f"Pre-processing started at: \t {dt.datetime.now()}")
        start_date_time = dt.datetime.now()
        self.start_date_time = start_date_time
        start_time = time()

        self.bore_id = bore_id
        if other_variables is None:
            other_variables = []

        out_after = 1  # time leads --> Periods ahead
        use_newest_data = True
        fix_to_calendar = False

        kernel = 'rbf'  # 'linear', 'rbf', 'sigmoid', 'precomputed'
        gamma = 'scale'
        epsilon = 0.1
        tolerance = 0.00001
        reg_parameter = 1.0
        degree = 3  # For polynomial kernel --> Ignored otherwise
        coef0 = 0.0  # only for ploy and sigmoid
        svr_shrink = True
        svr_scoring_code = 'neg_mean_squared_error'  # 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2', 'max_error'

        scaler_type = 'mm'  # 'mm', 'ss'
        test_size = 0.2
        shuffle_order = False

        optimiser = 'adam'
        loss = 'mse'  # 'mse', 'mae'
        metric = RMSE()  # 'accuracy', RMSE()
        epochs = 100
        keras_validation_split = 0.2
        verbose = 0
        # learning_rate = 0.001

        lstm_dropout_rate = 0.2
        lstm_recurrent_dropout_rate = 0.2
        lstm_layers = 1
        fch_layers = 2  # Fully Connected Hidden Layers
        num_lstm_cells = 64
        num_fc_neurons = 32

        interpolation_method = 'Linear'

        # Clear temp folder
        import_functions.clear_temp_folder(bore_id)

        # store raw data about the bore
        bore = bore_class.bore(bore_id, self.use_awo, years_of_data, use_newest_data, fix_to_calendar)

        # class for things used by the model
        model_parameters = model_parameters_class.model_parameters(epochs, periods_in, out_after, other_variables, av_period, gwl_input, gwl_output)

        # import data
        bore.add_dfs()  # adds gwl_df, bore_df, swl_df, awo_df
        bore.handle_missing_dates(interpolation_method)

        # remove unused dates from the imported data
        bore.remove_null_dates()

        # add reference (unscaled) bore data to bore for reference
        bore.add_silo_data()
        if self.use_awo:
            bore.add_awo_data()
        bore.add_gwl_data(out_after)
        bore.average_data(av_period)
        bore.add_data_dict()

        # format data for training
        model_parameters.add_data(bore)
        model_parameters.scale_data(scaler_type)
        model_parameters.format_inputs_outputs()
        model_parameters.divide_data(test_size, shuffle_order)

        # Dependent Variables
        input_shape = model_parameters.X_train.shape
        print(f"Pre-processing completed at: \t {dt.datetime.now()}")

        # LSTM Model
        keras_model = keras_lstm_class.keras_LSTM(optimiser=optimiser, loss=loss, metric=metric, epochs=epochs, keras_validation_split=keras_validation_split, verbose=verbose)
        keras_model.add_sets(model_parameters)
        keras_model.create_network(input_shape, num_lstm_cells, num_fc_neurons, lstm_dropout_rate, lstm_recurrent_dropout_rate, lstm_layers, fch_layers)
        print(f"LSTM training started at: \t {dt.datetime.now()}")
        keras_model.train_model(model_parameters.X_train, model_parameters.y_train)
        print(f"LSTM training completed at: \t {dt.datetime.now()}")

        keras_model.predict(model_parameters.X_test, scaler=model_parameters.output_scaler, dataset='test')
        keras_model.predict(model_parameters.X_train, scaler=model_parameters.output_scaler, dataset='train')
        score_functions.calculate_scores(keras_model)

        # SVR Model
        model_parameters.format_for_sklearn_SVR()
        sklearn_model = sklearn_svr_class.sklearn_SVR(kernel=kernel, gamma=gamma, epsilon=epsilon, C=reg_parameter, tolerance=tolerance, degree=degree, coef0=coef0, shrinking=svr_shrink, verbose=verbose, shuffle_order=shuffle_order, epochs=epochs)
        sklearn_model.add_sets(model_parameters)
        sklearn_model.create_model()
        print(f"SVR training started at:   \t {dt.datetime.now()}")
        sklearn_model.train_model(model_parameters.X_train_svr, model_parameters.y_train_svr, svr_scoring_code)
        print(f"SVR training completed at: \t {dt.datetime.now()}")

        sklearn_model.predict(model_parameters.X_test_svr, scaler=model_parameters.output_scaler, dataset='test')
        sklearn_model.predict(model_parameters.X_train_svr, scaler=model_parameters.output_scaler, dataset='train')
        score_functions.calculate_scores(sklearn_model)

        end_time = time()

        # For testing in the console
        self.mp = model_parameters
        self.br = bore
        self.km = keras_model
        self.sm = sklearn_model

        print(f"Output log started at:    \t {dt.datetime.now()}")
        output_summary = output_summary_class.output_summary(start_date_time, start_time, end_time, bore, model_parameters, keras_model, sklearn_model)
        output_summary.create_general_text()
        output_summary.create_input_variable_graphs()
        output_summary.create_lstm_text()
        output_summary.create_lstm_learning_graphs()
        output_summary.create_result_graphs(keras_model)
        output_summary.create_svr_text()
        output_summary.create_svr_learning_graphs()
        output_summary.create_result_graphs(sklearn_model)
        output_summary.create_log()
        output_summary.create_spreadsheet()
        output_summary.save_models()
        print(f"Output log completed at:  \t {dt.datetime.now()}")


if __name__ == '__main__':
    bore_id = 'GW080415.1.1'  # 'GW036872.1.1', 'GW075025.1.1', 'GW075405.1.1', 'GW080079.1.1' 'GW080415.1.1', 'GW080980.1.1', 'GW081101.1.2', 'GW273314.1.1', 'GW403746.3.3'

    gwl_output = 'average_delta'  # 'standard', 'delta', 'average', 'average_delta'
    gwl_input = 'average_delta'  # 'standard', 'delta', 'none', 'average', 'average_delta'
    other_variables = ['av_daily_rain', 'av_max_temp', 'av_vp', 'av_evap_syn', 'av_radiation', 'av_rh_tmin', 'av_et_morton_actual', 'av_mslp']

    years_of_data = 'all'  # 'all' or number
    av_period = 30  # days to average
    periods_in = 3  # time lags

    main_script_once = main_script(use_awo=False)
    main_script_once.run(bore_id, gwl_output, gwl_input, other_variables, years_of_data, av_period, periods_in)


# other_variables = ['daily_rain', 'max_temp', 'min_temp', 'vp', 'vp_deficit',
    # 'evap_pan', 'evap_syn', 'evap_comb', 'evap_morton_lake',
    # 'radiation', 'rh_tmax', 'rh_tmin', 'et_short_crop',
    # 'et_tall_crop', 'et_morton_actual', 'et_morton_potential',
    # 'et_morton_wet','mslp', 'sm_pct', 's0_pct', 'ss_pct', 'sd_pct', 'dd',
    # # Averages below here
    # 'av_daily_rain', 'av_max_temp', 'av_min_temp', 'av_vp', 'av_vp_deficit',
    # 'av_evap_pan', 'av_evap_syn', 'av_evap_comb', 'av_evap_morton_lake',
    # 'av_radiation', 'av_rh_tmax', 'av_rh_tmin', 'av_et_short_crop',
    # 'av_et_tall_crop', 'av_et_morton_actual', 'av_et_morton_potential',
    # 'av_et_morton_wet', 'av_mslp', 'av_sm_pct', 'av_s0_pct', 'av_ss_pct', 'av_sd_pct', 'av_dd'] #sm = root zone, s0 = upper, ss = lower, sd = deep layer, dd = deep drainage

