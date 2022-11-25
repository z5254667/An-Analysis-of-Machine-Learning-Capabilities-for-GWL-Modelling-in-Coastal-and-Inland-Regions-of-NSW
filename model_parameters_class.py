#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:45:30 2022

@author: johnsalvaris
"""

# Import packages
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # min-max to 0-1
from sklearn.preprocessing import StandardScaler  # z-score


class model_parameters():
    def __init__(self, epochs, periods_in, out_after, other_variables, av_period=30, gwl_input='standard', gwl_output='standard'):
        if np.all(gwl_output != np.array(['standard', 'delta', 'average', 'average_delta'])):
            raise Exception("gwl_output should be 'standard', 'delta', 'average' or 'average_delta'.")

        if np.all(gwl_input != np.array(['standard', 'delta', 'average', 'average_delta', 'none'])):
            raise Exception("gwl_input should be 'standard', 'delta', 'average', 'average_delta' or 'none'.")

        if out_after < 1:
            raise Exception("Output must be at least one day after final input. Change out_after")

        self.num_epochs = epochs
        self.periods_in = periods_in  # number of input periods
        self.out_after = out_after  # periods to first predicition e.g. 7 for one week in the future if period length is one day
        self.av_period = av_period  # Length of period (in days) which gets averaged
        self.input_variables = other_variables.copy()
        self.gwl_input = gwl_input
        self.gwl_output = gwl_output

        input_code_dict = {'standard': 'swl',
                           'delta': 'daily_delta_swl',
                           'average': 'av_swl',
                           'average_delta': 'av_period_delta_swl'}
        output_code_dict = {'standard': 'swl',
                            'delta': 'delta_swl',
                            'average': 'av_swl',
                            'average_delta': 'av_delta_swl'}

        if self.gwl_input != 'none':
            self.input_variables.insert(0, input_code_dict[self.gwl_input])

        self.output_variable = output_code_dict[self.gwl_output]


    def add_data(self, bore):
        if len(self.input_variables) == 0:
            raise Exception("No inputs assigned.")

        check_if_average_variables = sum(list(map(lambda x: 1 if (x[0] == 'a') else 0, self.input_variables)))
        if self.output_variable[0] == 'a':
            check_if_average_variables += 1

        if check_if_average_variables != len(self.input_variables) + 1 and check_if_average_variables != 0:
            raise Exception("Inputs and outputs must be consistently averages or consistently not averages.")

        def make_same_size(df_a, df_b):
            # If df_a longer than df_b, make df_a the same size as df_b by removing extra entries at the start
            df_a = df_a[-len(df_b):]
            df_a.reset_index(drop=True, inplace=True)

            # If df_b longer than df_a, make df_b the same size as df_a by removing extra entries at the start
            df_b = df_b[-len(df_a):]
            df_b.reset_index(drop=True, inplace=True)

            return df_a, df_b

        # Input Data
        for i in range(len(self.input_variables)):
            if i == 0:
                self.input_data = pd.DataFrame({self.input_variables[i]: bore.data_dict[self.input_variables[i]]})
            else:
                temp_df = pd.DataFrame({self.input_variables[i]: bore.data_dict[self.input_variables[i]]})
                temp_df, self.input_data = make_same_size(temp_df, self.input_data)
                self.input_data = pd.concat((self.input_data, temp_df), axis=1)

        # Assign correct date list based on inputs
        if check_if_average_variables != 0:
            self.input_dates = pd.DataFrame({'date': bore.av_dates})
        else:
            bore.av_period = 1
            self.input_dates = pd.DataFrame({'date': bore.dates})

        # Define output data
        self.output_data = pd.DataFrame({self.output_variable: bore.data_dict[self.output_variable]})

        # Make input output and dates the same size
        self.input_data, self.output_data = make_same_size(self.input_data, self.output_data)
        self.input_data, self.input_dates = make_same_size(self.input_data, self.input_dates)
        self.input_dates, self.output_data = make_same_size(self.input_dates, self.output_data)

        # Real swl values for use if delta is output
        if self.gwl_output in ['standard', 'delta']:
            self.output_swl = pd.DataFrame({'output_swl': bore.data_dict['swl']})
            self.output_current_swl = pd.DataFrame({'output_current_swl': bore.data_dict['swl']})
        else:
            self.output_swl = pd.DataFrame({'output_swl': bore.data_dict['av_swl']})
            self.output_current_swl = pd.DataFrame({'output_current_swl': bore.data_dict['av_swl']})

        # Remove dates not corresponding to an output
        self.output_data = self.output_data[self.periods_in + self.out_after - 1:]
        self.output_data.reset_index(drop=True, inplace=True)

        self.output_dates = self.input_dates[self.periods_in + self.out_after - 1:]
        self.output_dates.reset_index(drop=True, inplace=True)

        self.output_swl, self.output_data = make_same_size(self.output_swl, self.output_data)  # length is already correct

        self.output_current_swl, self.input_data = make_same_size(self.output_current_swl, self.input_data)
        self.output_current_swl = self.output_current_swl[self.periods_in - 1:-self.out_after]
        self.output_current_swl.reset_index(drop=True, inplace=True)


    def scale_data(self, scaler_type='mm'):
        if scaler_type == 'mm':
            self.input_scaler = MinMaxScaler()
            self.output_scaler = MinMaxScaler()
        elif scaler_type == 'ss':
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
        else:
            raise Exception("Invalid scaler_type choice. Must be 'mm' or 'ss'")

        self.input_scaler = self.input_scaler.fit(self.input_data[self.input_data.columns])
        self.output_scaler = self.output_scaler.fit(self.output_data.values.reshape(-1, 1))

        self.scaled_input = pd.DataFrame(self.input_scaler.transform(self.input_data[self.input_data.columns]))
        self.scaled_input.columns = list(self.input_data.columns)
        self.scaled_output = pd.DataFrame(self.output_scaler.transform(self.output_data.values.reshape(-1, 1)))


    def format_inputs_outputs(self):
        self.num_samples = self.scaled_output.shape[0]

        # format inputs
        input_list = []
        for i in range(self.num_samples):
            input_list.append(self.scaled_input.shift(-i, axis=0)[:self.periods_in].to_numpy())
        self.formatted_inputs = np.array(input_list)

        # format outputs
        self.formatted_outputs = self.scaled_output.to_numpy()


    def divide_data(self, test_size=0.2, shuffle=False, keras_validation_split=0):
        self.test_size = test_size  # % of dataset for testing
        self.shuffle = shuffle

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.formatted_inputs, self.formatted_outputs, test_size=self.test_size, shuffle=self.shuffle)
        self.train_dates, self.test_dates = train_test_split(self.output_dates, test_size=self.test_size, shuffle=self.shuffle)
        self.train_dates.reset_index(drop=True, inplace=True)
        self.test_dates.reset_index(drop=True, inplace=True)
        self.train_samples, self.test_samples = len(self.train_dates), len(self.test_dates)
        self.train_output_swl, self.test_output_swl = train_test_split(self.output_swl, test_size=self.test_size, shuffle=self.shuffle)
        self.train_output_current_swl, self.test_output_current_swl = train_test_split(self.output_current_swl, test_size=self.test_size, shuffle=self.shuffle)

        self.all_dates = np.concatenate([self.train_dates, self.test_dates])
        self.unscaled_y_train, self.unscaled_y_test = train_test_split(self.output_data, test_size=self.test_size, shuffle=self.shuffle)


    def format_for_sklearn_SVR(self):
        if self.X_train.ndim == 3:
            self.X_train_svr = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1] * self.X_train.shape[2]))
            self.X_test_svr = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1] * self.X_test.shape[2]))
        else:
            raise Exception('X_train not in format [samples, timesteps, features]')
        if self.y_train.shape[1] >= 2:
            raise Exception('SVR only capable of making one prediction at a time')
        else:
            self.y_train_svr = self.y_train.reshape(self.y_train.shape[0])
            self.y_test_svr = self.y_test.reshape(self.y_test.shape[0])


    """
    If time to create validation split function modify below:

    def val(x,y,val_split):
        split_at = int(x.shape[0] * (1. - val_split))
        X_validate = x[split_at:]
        X_train = x[:split_at]
        y_validate = y[split_at: ]
        y_train = y[:split_at]
        return

    """
