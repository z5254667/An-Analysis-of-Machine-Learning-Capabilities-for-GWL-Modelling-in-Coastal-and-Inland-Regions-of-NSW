#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:51:31 2022

@author: johnsalvaris
"""

# Import created scripts
import import_functions

# Import packages
import itertools
import math
import pickle
import PyPDF2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fpdf import FPDF
from functools import reduce
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf


class output_summary():
    def __init__(self, start_date_time, start_time, end_time, bore, model_parameters, keras_model, sklearn_model):
        self.start_time = start_time
        self.end_time = end_time
        self.start_date_time = start_date_time
        self.bore = bore
        self.model_parameters = model_parameters
        self.keras_model = keras_model
        self.sklearn_model = sklearn_model
        self.bore_id = bore.id
        self.inputs_dict = {'swl': 'Standing Water Level (m)',
                            'daily_rain': 'Rainfall (mm)',
                            'max_temp': 'Maximum Temperature (°C)',
                            'min_temp': 'Minimum Temperature (°C)',
                            'vp': 'Vapour Pressure (hPa)',
                            'vp_deficit': 'Vapour Pressure Deficit (hPa)',
                            'evap_pan': 'Evaporation - Class A Pan (mm)',
                            'evap_syn': 'Evaporation - Synthetic Estimate (mm)',
                            'evap_comb': 'Evaporation - Combination (Synthetic Estimate pre-1970, Class A Pan 1970 Onwards) (mm)',
                            'evap_morton_lake': "Evaporation - Morton's Shallow Lake Evaporation (mm)",
                            'radiation': 'Solar Radiation - Total Incoming Downward Shortwave Radiation on a Horizontal Surface (MJ/m^2)',
                            'rh_tmax': 'Relative Humidity at Time of Maximum Temperature (%)',
                            'rh_tmin': 'Relative Humidity at Time of Minimum Temperature (%)',
                            'et_short_crop': 'Evapotranspiration - FAO56 Short Crop (mm)',
                            'et_tall_crop': 'Evapotranspiration - ASCE Tall Crop (mm)',
                            'et_morton_actual': "Evapotranspiration - Morton's Areal Actual Evapotranspiration (mm)",
                            'et_morton_potential': "Evapotranspiration - Morton's Potential Evapotranspiration (mm)",
                            'et_morton_wet': "Evapotranspiration - Wet-Environment Areal Evapotranspiration Over Land (mm)",
                            'mslp': 'Mean Sea Level Pressure (hPa)',
                            'delta_swl': 'Change in Standing Water Level (m)',
                            'daily_delta_swl': 'Daily Change in Standing Water Level (m)',
                            'sm_pct': 'Absolute Root Zone Soil Moisture (0-100cm) (%)',
                            's0_pct': 'Absolute Upper Layer Soil Moisture (0-10cm) (%)',
                            'ss_pct': 'Absolute Lower Layer Soil Moisture (10-100cm) (%)',
                            'sd_pct': 'Absolute Deep Layer Soil Moisture (1-6m) (%)',
                            'dd': 'Absolute Deep Drainage below 6m (mm)',
                            'av_swl': 'Average Standing Water Level (m)',
                            'av_daily_rain': 'Average Rainfall (mm)',
                            'av_max_temp': 'Average Maximum Temperature (°C)',
                            'av_min_temp': 'Average Minimum Temperature (°C)',
                            'av_vp': 'Average Vapour Pressure (hPa)',
                            'av_vp_deficit': 'Average Vapour Pressure Deficit (hPa)',
                            'av_evap_pan': 'Average Evaporation - Class A Pan (mm)',
                            'av_evap_syn': 'Average Evaporation - Synthetic Estimate (mm)',
                            'av_evap_comb': 'Average Evaporation - Combination (Synthetic Estimate pre-1970, Class A Pan 1970 Onwards) (mm)',
                            'av_evap_morton_lake': "Average Evaporation - Morton's Shallow Lake Evaporation (mm)",
                            'av_radiation': 'Average Solar Radiation - Total Incoming Downward Shortwave Radiation on a Horizontal Surface (MJ/m^2)',
                            'av_rh_tmax': 'Average Relative Humidity at Time of Maximum Temperature (%)',
                            'av_rh_tmin': 'Average Relative Humidity at Time of Minimum Temperature (%)',
                            'av_et_short_crop': 'Average Evapotranspiration - FAO56 Short Crop (mm)',
                            'av_et_tall_crop': 'Average Evapotranspiration - ASCE Tall Crop (mm)',
                            'av_et_morton_actual': "Average Evapotranspiration - Morton's Areal Actual Evapotranspiration (mm)",
                            'av_et_morton_potential': "Average Evapotranspiration - Morton's Potential Evapotranspiration (mm)",
                            'av_et_morton_wet': "Average Evapotranspiration - Wet-Environment Areal Evapotranspiration Over Land (mm)",
                            'av_mslp': 'Average Mean Sea Level Pressure (hPa)',
                            'av_delta_swl': 'Average Change in Standing Water Level (m)',
                            'av_period_delta_swl': 'Average Period Change in Standing Water Level (m)',
                            'av_sm_pct': 'Average Absolute Root Zone Soil Moisture (0-100cm) (%)',
                            'av_s0_pct': 'Average Absolute Upper Layer Soil Moisture (0-10cm) (%)',
                            'av_ss_pct': 'Average Absolute Lower Layer Soil Moisture (10-100cm) (%)',
                            'av_sd_pct': 'Average Absolute Deep Layer Soil Moisture (1-6m) (%)',
                            'av_dd': 'Average Absolute Deep Drainage (below 6m) (mm)'}

        # Create Files
        self.temp_general_text_path = import_functions.make_path(f'{self.bore_id}/temp', 'temp_general_text.pdf')
        self.temp_general_text_file = FPDF()
        self.temp_general_text_file.add_page()
        self.temp_general_text_file.set_font('Courier', size=8)

        self.temp_input_graphs_path_1 = import_functions.make_path(f'{self.bore_id}/temp', 'temp_input_graphs_1.pdf')
        self.temp_input_graphs_file_1 = PdfPages(self.temp_input_graphs_path_1)

        self.temp_input_graphs_path_2 = import_functions.make_path(f'{self.bore_id}/temp', 'temp_input_graphs_2.pdf')
        self.temp_input_graphs_file_2 = PdfPages(self.temp_input_graphs_path_2)

        self.temp_input_graphs_path_3 = import_functions.make_path(f'{self.bore_id}/temp', 'temp_input_graphs_3.pdf')
        self.temp_input_graphs_file_3 = PdfPages(self.temp_input_graphs_path_3)

        self.temp_input_graphs_path_4 = import_functions.make_path(f'{self.bore_id}/temp', 'temp_input_graphs_4.pdf')
        self.temp_input_graphs_file_4 = PdfPages(self.temp_input_graphs_path_4)

        self.temp_lstm_text_path = import_functions.make_path(f'{self.bore_id}/temp', 'temp_lstm_text.pdf')
        self.temp_lstm_text_file = FPDF()
        self.temp_lstm_text_file.add_page()
        self.temp_lstm_text_file.set_font('Courier', size=8)

        self.temp_lstm_learn_graphs_path = import_functions.make_path(f'{self.bore_id}/temp', 'temp_lstm_learn_graphs.pdf')
        self.temp_lstm_learn_graphs_file = PdfPages(self.temp_lstm_learn_graphs_path)

        self.temp_lstm_results_graphs_path = import_functions.make_path(f'{self.bore_id}/temp', 'temp_lstm_results_graphs.pdf')
        self.temp_lstm_results_graphs_file = PdfPages(self.temp_lstm_results_graphs_path)

        self.temp_svr_results_graphs_path = import_functions.make_path(f'{self.bore_id}/temp', 'temp_svr_results_graphs.pdf')
        self.temp_svr_results_graphs_file = PdfPages(self.temp_svr_results_graphs_path)

        self.temp_svr_text_path = import_functions.make_path(f'{self.bore_id}/temp', 'temp_svr_text.pdf')
        self.temp_svr_text_file = FPDF()
        self.temp_svr_text_file.add_page()
        self.temp_svr_text_file.set_font('Courier', size=8)

        self.temp_svr_learn_graphs_path = import_functions.make_path(f'{self.bore_id}/temp', 'temp_svr_learn_graphs.pdf')
        self.temp_svr_learn_graphs_file = PdfPages(self.temp_svr_learn_graphs_path)


    def create_general_text(self):
        # Define Variables
        run_time = round(self.end_time - self.start_time, 3)
        start_date_time_string = self.start_date_time.strftime("%d/%m/%Y - %H:%M:%S")
        region = self.bore.region
        bore_coordinates = f'({self.bore.bore_latitude}, {self.bore.bore_longitude})'
        silo_coordinates = f'({self.bore.silo_latitude}, {self.bore.silo_longitude})'
        agency = self.bore.bore_df["Agency"][0]
        drilled_date = self.bore.bore_df["Drilled Date"][0]
        drilled_date = f"{drilled_date[8:10]}/{drilled_date[5:7]}/{drilled_date[:4]}"
        bore_depth = self.bore.bore_df["Bore Depth (m)"][0]
        drilled_depth = self.bore.bore_df["Drilled Depth (m)"][0]
        reference_elevation = self.bore.bore_df["Reference Elevation (m AHD)"][0]
        time_series_reference_elevation = self.bore.bore_df["Time Series Reference Elevation (m AHD)"][0]
        land_surface_elevation = self.bore.bore_df["Land Surface Elevation"][0]
        average_period = self.bore.av_period
        output_variable = self.inputs_dict[self.model_parameters.output_variable]
        out_after = self.model_parameters.out_after
        interpolation_method = self.bore.interpolation_method
        self.years_of_data = self.bore.years_of_data
        if self.years_of_data == 'all':
            self.years_of_data = f'{round(len(self.bore.swl) / 365, 4)} (All Data Available)'

        start_date_string = str(self.model_parameters.input_dates['date'].to_numpy()[0])
        start_date_year = start_date_string[:4]
        start_date_month = start_date_string[5:7]
        start_date_day = start_date_string[8:10]

        end_date_string = str(self.model_parameters.input_dates['date'].to_numpy()[-1])
        end_date_year = end_date_string[:4]
        end_date_month = end_date_string[5:7]
        end_date_day = end_date_string[8:10]

        self.data_start = f'{start_date_day}/{start_date_month}/{start_date_year}'
        self.data_end = f'{end_date_day}/{end_date_month}/{end_date_year}'

        self.train_set_percentage = f'{(1 - self.model_parameters.test_size) * 100}%'
        self.test_set_percentage = f'{self.model_parameters.test_size * 100}%'

        periods_before = self.model_parameters.periods_in - 1

        quality_code = np.array(list(map(lambda x: x[-1], self.bore.gwl_df['Quality Code'])))
        unique, counts = np.unique(quality_code, return_counts=True)
        percents = counts / np.sum(counts) * 100
        quality_summary = pd.DataFrame(dict(zip(unique, [[c, p] for c, p in zip(counts, percents)])))
        quality_summary.index = ['Number', 'Percentage (%)']

        # File Text
        lines = []
        lines.append('<><> Time Stamp <><>')
        lines.append('')
        lines.append(f'Code Started: {start_date_time_string}')
        lines.append(f'Total Run Time: {run_time} s')
        lines.append('')
        lines.append('')

        lines.append('<><> Bore Information <><>')
        lines.append('')
        lines.append(f'Bore ID: {self.bore_id}')
        lines.append(f'Region: {region}')
        lines.append(f'Bore Coordinates: {bore_coordinates}')
        lines.append(f'Agency: {agency}')
        lines.append(f'Drilled Date: {drilled_date}')
        lines.append(f'Bore Depth: {bore_depth} m')
        lines.append(f'Drilled Depth: {drilled_depth} m')
        lines.append(f'Reference Elevation: {reference_elevation} m')
        lines.append(f'Time Series Reference Elevation: {time_series_reference_elevation} m')
        lines.append(f'Land Surface Elevation: {land_surface_elevation} m')
        lines.append(f'SILO Grid Point Coordinates: {silo_coordinates}')
        lines.append('')
        lines.append('')

        lines.append('<><> Model Output <><>')
        lines.append('')
        lines.append(f'Averaged Period: {average_period} day(s)')
        lines.append(f'Output: {output_variable} in {out_after} period(s) time')
        lines.append('')
        lines.append('')

        lines.append('<><> Model Inputs <><>')
        lines.append('')
        lines.append(f'Years of Data Provided: {self.years_of_data}')
        lines.append(f'Data Range: {self.data_start} - {self.data_end}')
        lines.append(f'Train Set Size: {self.train_set_percentage}')
        lines.append(f'Test Set Size: {self.test_set_percentage}')
        lines.append(f'Train Set Samples: {self.model_parameters.train_samples}')
        lines.append(f'Test Set Samples: {self.model_parameters.test_samples}')
        lines.append(f'Input Timesteps: Current period + {periods_before} preceeding period(s)')
        lines.append('Input Variables:')
        for iv in self.model_parameters.input_variables:
            variable_name = self.inputs_dict[iv]
            lines.append(f'    {variable_name}')
        lines.append('')
        lines.append('')

        lines.append('<><> Data Quality <><>')
        lines.append('')
        lines.append(f'Interpolation Method: {interpolation_method}')
        for q in quality_summary:
            number = int(quality_summary[q]['Number'])
            percent = round(quality_summary[q]['Percentage (%)'], 2)
            lines.append(f"Quality Code: {q}, \t Number: {number}, \t Percentage: {percent}%")

        # Add text to file
        for line in lines:
            self.temp_general_text_file.cell(0, 5, txt=line, ln=1, align='L')

        self.temp_general_text_file.output(self.temp_general_text_path)


    def create_input_variable_graphs(self):
        # Define variables
        num_inputs = len(self.model_parameters.input_variables)
        input_dates = self.model_parameters.input_dates.to_numpy()
        input_variables = self.model_parameters.input_variables
        input_data = self.model_parameters.input_data
        pages = math.ceil(num_inputs / 10)  # Maximum 10 graphs per page
        periods_in = self.model_parameters.periods_in
        combinations = list(itertools.combinations(input_variables, 2))
        num_combinations = len(combinations)

        # Calibrate plot settings
        sns.set_context(rc={'lines.linewidth': 0.5, 'lines.markersize': 0.5})

        # Timeseries Plots
        num_plotted = 0
        for p in range(pages):
            if p == pages - 1:
                plots_per_page = num_inputs - math.ceil(num_inputs / pages) * p
            else:
                plots_per_page = math.ceil(num_inputs / pages)

            fig, ax = plt.subplots(plots_per_page, 1, figsize=(8.27, 11.69))

            for i in range(plots_per_page):
                k = num_plotted + i

                if len(self.inputs_dict[input_variables[k]]) >= 90:
                    font_size = 9
                else:
                    font_size = 10

                if plots_per_page == 1:
                    ax.plot(input_dates, input_data[input_variables[k]])
                    ax.set_title(self.inputs_dict[input_variables[k]], fontsize=font_size)
                else:
                    ax[i].plot(input_dates, input_data[input_variables[k]])
                    ax[i].set_title(self.inputs_dict[input_variables[k]], fontsize=font_size)

            fig.tight_layout()
            self.temp_input_graphs_file_1.savefig(fig)
            plt.close()
            num_plotted += plots_per_page
        self.temp_input_graphs_file_1.close()

        # ACF/PACF Plots
        num_plotted = 0
        for p in range(pages):
            if p == pages - 1:
                plots_per_page = num_inputs - math.ceil(num_inputs / pages) * p
            else:
                plots_per_page = math.ceil(num_inputs / pages)

            fig, ax = plt.subplots(plots_per_page, 2, figsize=(8.27, 11.69))

            for i in range(plots_per_page):
                k = num_plotted + i  # Input variable index

                if len(self.inputs_dict[input_variables[k]] + ' - Partial Autocorrelation') >= 75:
                    font_size = 4
                else:
                    font_size = 7

                if plots_per_page == 1:
                    plot_acf(input_data[input_variables[k]], lags=periods_in, ax=ax[0], markersize=3)
                    plot_pacf(input_data[input_variables[k]], lags=periods_in, ax=ax[1], method='ywm', markersize=3)
                    ax[0].set_title(self.inputs_dict[input_variables[k]] + ' - Autocorrelation', fontsize=font_size)
                    ax[1].set_title(self.inputs_dict[input_variables[k]] + ' - Partial Autocorrelation', fontsize=font_size)
                else:
                    plot_acf(input_data[input_variables[k]], lags=periods_in, ax=ax[i, 0], markersize=3)
                    plot_pacf(input_data[input_variables[k]], lags=periods_in, ax=ax[i, 1], method='ywm', markersize=3)
                    ax[i, 0].set_title(self.inputs_dict[input_variables[k]] + ' - Autocorrelation', fontsize=font_size)
                    ax[i, 1].set_title(self.inputs_dict[input_variables[k]] + ' - Partial Autocorrelation', fontsize=font_size)

            fig.tight_layout()
            self.temp_input_graphs_file_2.savefig(fig)
            plt.close()
            num_plotted += plots_per_page
        self.temp_input_graphs_file_2.close()

        # Correlation Plots (At least two variables only)
        if num_combinations >= 1:
            plt.rcParams['figure.dpi'] = 400

            # Scatter Plots
            def get_factors(n):
                step = 2 if n % 2 else 1
                return reduce(list.__add__, ([i, n // i] for i in range(1, int(math.sqrt(n)) + 1, step) if n % i == 0))

            factors = get_factors(num_combinations)
            ratio = np.array([])

            for i in range(1, len(factors), 2):
                ratio = np.append(ratio, abs(factors[i - 1] / factors[i] - 8.27 / 11.69))

            optimal = np.argmin(ratio) * 2
            optimal_rows = factors[optimal + 1]
            optimal_cols = factors[optimal]
            vertical_divisions = math.ceil(optimal_rows / 10)  # Maxmimum 10 rows of graphs per page
            horizontal_divisions = math.ceil(optimal_cols / 7)  # Maximum 7 columns of graphs per page

            num_plotted = 0
            for p in range(vertical_divisions):
                if p == vertical_divisions - 1:
                    page_rows = optimal_rows - math.ceil(optimal_rows / vertical_divisions) * p
                else:
                    page_rows = math.ceil(optimal_rows / vertical_divisions)

                for q in range(horizontal_divisions):
                    if q == horizontal_divisions - 1:
                        page_cols = optimal_cols - math.ceil(optimal_cols / horizontal_divisions) * q
                    else:
                        page_cols = math.ceil(optimal_cols / horizontal_divisions)

                    plots_per_page = page_rows * page_cols
                    if plots_per_page < 36:
                        sns.set_context(rc={'lines.linewidth': 0.5, 'lines.markersize': 0.5, 'font.size': 10, 'axes.linewidth': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3})
                    else:
                        sns.set_context(rc={'lines.linewidth': 0.5, 'lines.markersize': 0.5, 'font.size': 7, 'axes.linewidth': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3})

                    fig, ax = plt.subplots(page_rows, page_cols, figsize=(8.27, 11.69))
                    for i in range(page_rows):
                        for j in range(page_cols):
                            k = num_plotted + i * page_cols + j  # Input variable index

                            if page_rows == 1 and page_cols == 1:
                                sns.regplot(x=input_data[combinations[k][0]], y=input_data[combinations[k][1]], line_kws={'color': 'tab:orange'})
                                plt.xlabel(combinations[k][0])
                                plt.ylabel(combinations[k][1])
                            elif page_rows == 1 or page_cols == 1:
                                sns.regplot(x=input_data[combinations[k][0]], y=input_data[combinations[k][1]], ax=ax[k], line_kws={'color': 'tab:orange'})
                                ax[k].set_xlabel(combinations[k][0])
                                ax[k].set_ylabel(combinations[k][1])
                            else:
                                sns.regplot(x=input_data[combinations[k][0]], y=input_data[combinations[k][1]], ax=ax[i, j], line_kws={'color': 'tab:orange'})
                                ax[i, j].set_xlabel(combinations[k][0])
                                ax[i, j].set_ylabel(combinations[k][1])

                    num_plotted += plots_per_page
                    fig.suptitle('Input Variable Correlation')
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.96)

                    self.temp_input_graphs_file_3.savefig(fig)
                    plt.close()
            self.temp_input_graphs_file_3.close()

            # Heatmap
            if len(input_variables) < 6:
                sns.set_context(rc={'lines.linewidth': 0.5, 'lines.markersize': 0.5, 'font.size': 10, 'axes.linewidth': 0.8, 'xtick.major.size': 2, 'ytick.major.size': 2})
            elif len(input_variables) < 16:
                sns.set_context(rc={'lines.linewidth': 0.5, 'lines.markersize': 0.5, 'font.size': 7, 'axes.linewidth': 0.8, 'xtick.major.size': 2, 'ytick.major.size': 2})
            else:
                sns.set_context(rc={'lines.linewidth': 0.5, 'lines.markersize': 0.5, 'font.size': 4, 'axes.linewidth': 0.8, 'xtick.major.size': 2, 'ytick.major.size': 2})

            correlation_matrix = input_data.corr()
            fig, ax = plt.subplots(1, 1, figsize=(8.27, 11.69))
            sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap='RdBu', center=0)
            ax.set_title("Input Variable Correlation Heatmap", fontsize=10)

            self.temp_input_graphs_file_4.savefig(fig)
            plt.close()
            self.temp_input_graphs_file_4.close()
        else:
            self.temp_input_graphs_file_3.close()
            self.temp_input_graphs_file_4.close()

        # Reset Seaborn Styles
        sns.set_context(rc={'lines.linewidth': 0.5, 'lines.markersize': 0.5, 'font.size': 10, 'axes.linewidth': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3})


    def create_lstm_text(self):
        # Define variables
        optimiser = self.keras_model.optimiser
        loss = self.keras_model.loss
        metric = self.keras_model.metric
        epochs = self.keras_model.epochs
        validation_split = self.keras_model.val_split * 100
        verbose = self.keras_model.verbose
        input_shape = self.keras_model.full_input_shape
        lstm_cells = self.keras_model.lstm_cells
        fc_neurons = self.keras_model.fc_neurons
        num_outputs = self.keras_model.num_outputs
        dropout_rate = self.keras_model.lstm_dropout * 100
        recurrent_dropout_rate = self.keras_model.lstm_recurrent_dropout * 100
        summary_string = []
        self.keras_model.model.summary(print_fn=lambda x: summary_string.append(x))
        training_losses = self.keras_model.history.history['loss']
        train_scores = self.keras_model.train_scores_dict
        test_scores = self.keras_model.test_scores_dict
        lstm_layers = self.keras_model.lstm_layers
        fch_layers = self.keras_model.fch_layers

        # File text
        lines = []
        lines.append('<><> TensorFlow Keras LSTM Model <><>')
        lines.append('')
        lines.append(f'Optimiser: {optimiser}')
        lines.append(f'Loss: {loss}')
        lines.append(f'Metric: {metric}')
        lines.append(f'Number of Epochs: {epochs}')
        lines.append(f'Percentage of Training Data for Validation: {validation_split}%')
        if self.keras_model.shuffle_order:
            lines.append('Time Series Order: Shufffled')
        else:
            lines.append('Time Series Order: Chronological')
        if verbose == 0:
            lines.append('Verbose: Off')
        else:
            lines.append('Verbose: On')
        lines.append('')
        lines.append('')

        lines.append('<><> Model Architecture <><>')
        lines.append('')
        lines.append(f'Input Shape (Samples, Timesteps, Features): {input_shape}')
        lines.append(f'LSTM Layers: {lstm_layers}')
        lines.append(f'LSTM Cells per Layer: {lstm_cells}')
        lines.append(f'Fully Connected Hidden Layers: {fch_layers}')
        lines.append(f'Fully Connected Hidden Neurons per Layer: {fc_neurons}')
        lines.append(f'Fully Connected Output Neurons: {num_outputs}')
        lines.append(f'LSTM Dropout Rate: {dropout_rate}%')
        lines.append(f'LSTM Recurrent Dropout Rate: {recurrent_dropout_rate}%')
        lines.append('')
        for line in summary_string:
            lines.append(line)
        lines.append('')
        lines.append('')

        lines.append('<><> Training Loss <><>')
        lines.append('')
        for i in range(len(training_losses)):
            if len(training_losses) <= 10:
                lines.append(f"Epoch: {i + 1}, \t Loss: {training_losses[i]}")
            elif (i + 1) % (len(training_losses) // 10) == 0:
                lines.append(f"Epoch: {i + 1}, \t Loss: {training_losses[i]}")
        lines.append('')
        lines.append('')

        if validation_split > 0:
            validation_losses = self.keras_model.history.history['val_loss']
            lines.append('<><> Validation Loss <><>')
            lines.append('')
            for i in range(len(validation_losses)):
                if len(validation_losses) <= 10:
                    lines.append(f"Epoch: {i + 1}, \t Loss: {validation_losses[i]}")
                elif (i + 1) % (len(validation_losses) // 10) == 0:
                    lines.append(f"Epoch: {i + 1}, \t Loss: {validation_losses[i]}")
            lines.append('')
            lines.append('')

        def print_scores(scores_dict, output_text):
            for key, value in scores_dict.items():
                output_text.append(f'{key}: \t {round(scores_dict[key], 5)}')

        lines.append('<><> Training Set Scores <><>')
        lines.append('')
        print_scores(train_scores, lines)
        lines.append('')
        lines.append('')

        lines.append('<><> Test Set Scores <><>')
        lines.append('')
        print_scores(test_scores, lines)

        # Add text
        for line in lines:
            self.temp_lstm_text_file.cell(0, 5, txt=line, ln=1, align='L')
        self.temp_lstm_text_file.output(self.temp_lstm_text_path)


    def create_lstm_learning_graphs(self):
        # Define variables
        training_losses = self.keras_model.history.history['loss']
        validation_split = self.keras_model.val_split

        # Plot graphs
        plt.figure(figsize=(8.27, 11.69))
        plt.plot(training_losses)

        if validation_split > 0:
            validation_losses = self.keras_model.history.history['val_loss']
            plt.plot(validation_losses)
            plt.title('LSTM Learning Curves')
            plt.legend(['Training Loss', 'Validation Loss'])
        else:
            plt.title('LSTM Learning Curve')

        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.tight_layout()

        self.temp_lstm_learn_graphs_file.savefig()
        plt.close()
        self.temp_lstm_learn_graphs_file.close()


    def create_result_graphs(self, model_class):
        model_type = model_class.model_type
        active_file_dict = {'LSTM': self.temp_lstm_results_graphs_file, 'SVR': self.temp_svr_results_graphs_file}

        y_axis_label_dict = {'standard': 'SWL (m)', 'delta': 'Change in SWL (m)', 'average': 'Average SWL (m)', 'average_delta': 'Change in Average SWL (m)'}
        y_axis_label = y_axis_label_dict[self.model_parameters.gwl_output]

        train_dates = model_class.train_dates
        test_dates = model_class.test_dates
        all_dates = np.concatenate([train_dates, test_dates])

        train_actual = model_class.unscaled_y_train
        test_actual = model_class.unscaled_y_test
        all_actual = np.concatenate([train_actual, test_actual])

        train_predicted = model_class.unscaled_y_hat_train
        test_predicted = model_class.unscaled_y_hat_test
        all_predicted = np.concatenate([train_predicted, test_predicted])

        # Plot standard (Delta) SWL
        fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69))

        ax[0].axvline(x=train_dates[-1], c='r', linestyle='--')  # Marks separation between training and test set
        ax[0].plot(all_dates, all_actual, label='Actual')
        ax[0].plot(all_dates, all_predicted, label='Predicted')
        ax[0].set_ylabel(y_axis_label)
        ax[0].set_title(f'{model_type} Model: Training and Testing Sets')
        ax[0].legend()

        ax[1].plot(test_dates, test_actual, label='Actual')
        ax[1].plot(test_dates, test_predicted, label='Predicted')
        ax[1].set_ylabel(y_axis_label)
        ax[1].set_title(f'{model_type} Model: Testing Set')
        ax[1].legend()

        fig.tight_layout()
        active_file_dict[model_type].savefig(fig)
        plt.close()

        # Plot inverted (Delta) SWL
        fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69))

        ax[0].axvline(x=train_dates[-1], c='r', linestyle='--')  # Marks separation between training and test set
        ax[0].plot(all_dates, all_actual, label='Actual')
        ax[0].plot(all_dates, all_predicted, label='Predicted')
        ax[0].invert_yaxis()
        ax[0].set_ylabel(f'Inverted {y_axis_label}')
        ax[0].set_title(f'{model_type} Model: Training and Testing Sets')
        ax[0].legend()

        ax[1].plot(test_dates, test_actual, label='Actual')
        ax[1].plot(test_dates, test_predicted, label='Predicted')
        ax[1].invert_yaxis()
        ax[1].set_ylabel(f'Inverted {y_axis_label}')

        ax[1].legend()

        fig.tight_layout()
        active_file_dict[model_type].savefig(fig)
        plt.close()

        # Plot SWL Graphs if needed
        if self.model_parameters.gwl_output in ['delta', 'average_delta']:
            train_level_actual = model_class.train_output_swl
            test_level_actual = model_class.test_output_swl
            all_level_actual = np.concatenate([train_level_actual, test_level_actual])

            train_level_predicted = model_class.y_hat_level_train
            test_level_predicted = model_class.y_hat_level_test
            all_level_predicted = np.concatenate([train_level_predicted, test_level_predicted])

            y_axis_label = y_axis_label[10:]

            fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69))

            ax[0].axvline(x=train_dates[-1], c='r', linestyle='--')  # Marks separation between training and test set
            ax[0].plot(all_dates, all_level_actual, label='Actual')
            ax[0].plot(all_dates, all_level_predicted, label='Predicted')
            ax[0].set_ylabel(y_axis_label)
            ax[0].set_title(f'{model_type} Model: Training and Testing Sets')
            ax[0].legend()

            ax[1].plot(test_dates, test_level_actual, label='Actual')
            ax[1].plot(test_dates, test_level_predicted, label='Predicted')
            ax[1].set_ylabel(y_axis_label)
            ax[1].set_title(f'{model_type} Model: Testing Set')
            ax[1].legend()

            fig.tight_layout()
            active_file_dict[model_type].savefig(fig)
            plt.close()

            # Inverted Plot
            fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69))

            ax[0].axvline(x=train_dates[-1], c='r', linestyle='--')  # Marks separation between training and test set
            ax[0].plot(all_dates, all_level_actual, label='Actual')
            ax[0].plot(all_dates, all_level_predicted, label='Predicted')
            ax[0].invert_yaxis()
            ax[0].set_ylabel(f'Inverted {y_axis_label}')
            ax[0].set_title(f'{model_type} Model: Training and Testing Sets')
            ax[0].legend()

            ax[1].plot(test_dates, test_level_actual, label='Actual')
            ax[1].plot(test_dates, test_level_predicted, label='Predicted')
            ax[1].invert_yaxis()
            ax[1].set_ylabel(f'Inverted {y_axis_label}')
            ax[1].set_title(f'{model_type} Model: Testing Set')
            ax[1].legend()

            fig.tight_layout()
            active_file_dict[model_type].savefig(fig)
            plt.close()

        active_file_dict[model_type].close()


    def create_svr_text(self):
        # Define variables
        kernel = self.sklearn_model.kernel
        gamma = self.sklearn_model.gamma
        coef0 = self.sklearn_model.coef0
        degree = self.sklearn_model.degree
        epsilon = self.sklearn_model.epsilon
        tol = self.sklearn_model.tol
        c = self.sklearn_model.C
        shrinking = self.sklearn_model.shrinking
        verbose = self.sklearn_model.verbose
        num_sv = self.sklearn_model.model.n_support_[0]
        sv_size = self.sklearn_model.model.n_features_in_
        mean_train_score = self.sklearn_model.train_curve_scores_mean
        mean_validation_score = self.sklearn_model.test_curve_scores_mean
        train_scores = self.sklearn_model.train_scores_dict
        test_scores = self.sklearn_model.test_scores_dict

        # File text
        lines = []
        lines.append('<><> Scikit Learn SVR Model <><>')
        lines.append('')
        lines.append(f'Kernel Function: {kernel}')
        if kernel in ['rbf', 'poly', 'sigmoid']:
            lines.append(f'Kernel Coefficient: {gamma}')
        if kernel in ['poly', 'sigmoid']:
            lines.append(f'Independent Kernel Term: {coef0}')
        if kernel == 'poly':
            lines.append(f'Degree of Polynomial Kernel Function: {degree}')
        lines.append(f'Epsilon: {epsilon}')
        lines.append(f'Stopping Criterion Tolerance: {tol}')
        lines.append(f'Regularisation Parameter: {c}')
        lines.append(f'Shrinking: {shrinking}')
        if self.sklearn_model.shuffle:
            lines.append('Time Series Order: Shufffled')
        else:
            lines.append('Time Series Order: Chronological')
        if verbose == 0:
            lines.append('Verbose: Off')
        else:
            lines.append('Verbose: On')
        lines.append('')
        lines.append('')

        lines.append('<><> Model Architecture <><>')
        lines.append('')
        lines.append(f'Number of Support Vectors: {num_sv}')
        lines.append(f'Input/Support Vector Size: {sv_size}')
        lines.append('')
        lines.append('')

        lines.append('<><> 5-Fold Cross Validation Mean Training Loss <><>')
        lines.append('')
        for i in range(len(mean_train_score)):
            if len(mean_train_score) <= 10:
                lines.append(f"Epoch: {i + 1}, \t Loss: {mean_train_score[i]}")
            elif (i + 1) % (len(mean_train_score) // 10) == 0:
                lines.append(f"Epoch: {i + 1}, \t Loss: {mean_train_score[i]}")
        lines.append('')
        lines.append('')

        lines.append('<><> 5-Fold Cross Validation Mean Validation Loss <><>')
        lines.append('')
        for i in range(len(mean_validation_score)):
            if len(mean_validation_score) <= 10:
                lines.append(f"Epoch: {i + 1}, \t Loss: {mean_validation_score[i]}")
            elif (i + 1) % (len(mean_validation_score) // 10) == 0:
                lines.append(f"Epoch: {i + 1}, \t Loss: {mean_validation_score[i]}")
        lines.append('')
        lines.append('')

        def print_scores(scores_dict, output_text):
            for key, value in scores_dict.items():
                output_text.append(f'{key}: \t {round(scores_dict[key], 5)}')

        lines.append('<><> Training Set Scores <><>')
        lines.append('')
        print_scores(train_scores, lines)
        lines.append('')
        lines.append('')

        lines.append('<><> Test Set Scores <><>')
        lines.append('')
        print_scores(test_scores, lines)

        # Add text to file
        for line in lines:
            self.temp_svr_text_file.cell(0, 5, txt=line, ln=1, align='L')

        self.temp_svr_text_file.output(self.temp_svr_text_path)


    def create_svr_learning_graphs(self):
        # Define variables
        num_examples = self.sklearn_model.epoch_list
        mean_train_score = self.sklearn_model.train_curve_scores_mean
        train_std = self.sklearn_model.train_curve_scores_std
        mean_validation_score = self.sklearn_model.test_curve_scores_mean
        validation_std = self.sklearn_model.test_curve_scores_std
        scoring_code = self.sklearn_model.scoring_code
        mean_fit_time = self.sklearn_model.fit_times_mean
        fit_time_std = self.sklearn_model.fit_times_std

        fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69))

        # Plot learning scores vs epoch (number of examples)
        ax[0].set_title('SVR Learning Curve')
        ax[0].fill_between(num_examples, mean_train_score - train_std, mean_train_score + train_std, alpha=0.1)
        ax[0].fill_between(num_examples, mean_validation_score - validation_std, mean_validation_score + validation_std, alpha=0.1)
        ax[0].plot(num_examples, mean_train_score, label="Training score")
        ax[0].plot(num_examples, mean_validation_score, label="5-Fold Cross-validation score")
        ax[0].legend(loc="best")
        ax[0].set_xlabel("Training Examples")
        ax[0].set_ylabel(f"Average {scoring_code}")

        # Plot fit times vs epoch (training examples)
        ax[1].plot(num_examples, mean_fit_time)
        ax[1].fill_between(num_examples, mean_fit_time - fit_time_std, mean_fit_time + fit_time_std, alpha=0.1)
        ax[1].set_xlabel("Training Examples")
        ax[1].set_ylabel("Fit Times")
        ax[1].set_title("Model Scalability")

        plt.tight_layout()
        # plt.savefig(self.temp_svr_learn_graphs_file, format='pdf')
        self.temp_svr_learn_graphs_file.savefig(fig)
        plt.close()
        self.temp_svr_learn_graphs_file.close()


    def create_log(self):
        file_name_date_time = self.start_date_time.strftime("%Y_%m_%d__%H_%M_%S")

        log_file_name = f'output_log_{file_name_date_time}.pdf'
        log_file_path = import_functions.make_path(f'{self.bore_id}/logs', log_file_name)

        pdfs = [self.temp_general_text_path,
                self.temp_input_graphs_path_1,
                self.temp_input_graphs_path_2,
                self.temp_input_graphs_path_3,
                self.temp_input_graphs_path_4,
                self.temp_lstm_text_path,
                self.temp_lstm_learn_graphs_path,
                self.temp_lstm_results_graphs_path,
                self.temp_svr_text_path,
                self.temp_svr_learn_graphs_path,
                self.temp_svr_results_graphs_path]

        merger = PyPDF2.PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(log_file_path)
        merger.close()


    def create_spreadsheet(self):
        file_name_date_time = self.start_date_time.strftime("%Y_%m_%d__%H_%M_%S")

        spreadsheet_file_name = f'output_spreadsheet_{file_name_date_time}.xlsx'
        spreadsheet_file_path = import_functions.make_path(f'{self.bore_id}/spreadsheets', spreadsheet_file_name)

        input_dates = self.model_parameters.input_dates
        input_data = self.model_parameters.input_data
        output_dates = self.model_parameters.output_dates
        output_data = self.model_parameters.output_data
        quality = self.bore.gwl_df['Quality Code'].rename('quality')
        input_variables = self.model_parameters.input_variables
        output_current_swl = self.model_parameters.output_current_swl
        output_swl = self.model_parameters.output_swl
        scaled_input_data = self.model_parameters.scaled_input
        scaled_input_data.columns.columns = input_variables
        scaled_output_data = self.model_parameters.scaled_output
        scaled_output_data.columns = [self.model_parameters.output_variable]
        train_dates = self.model_parameters.train_dates
        test_dates = self.model_parameters.test_dates

        # General information sheet
        parameters = ['ID',
                      'Region',
                      'Latitude',
                      'Longitude',
                      'Years of Data Provided',
                      'Data Start',
                      'Data End',
                      'Average Period',
                      'Train Set Split',
                      'Test Set Split',
                      'Train Set Samples',
                      'Test Set Samples',
                      'Lag(s)',
                      'Lead(s)',
                      'Output']

        value = [self.bore_id,
                 self.bore.region,
                 self.bore.bore_latitude,
                 self.bore.bore_longitude,
                 self.years_of_data,
                 self.data_start,
                 self.data_end,
                 self.bore.av_period,
                 self.train_set_percentage,
                 self.test_set_percentage,
                 self.model_parameters.train_samples,
                 self.model_parameters.test_samples,
                 self.model_parameters.periods_in,
                 self.model_parameters.out_after,
                 self.model_parameters.output_variable]

        for i in range(len(self.model_parameters.input_variables)):
            parameters.append(f'Input {i+1}')
            value.append(self.model_parameters.input_variables[i])

        general_info_df = pd.DataFrame({'Parameters': value}, index=parameters)

        # Input data sheet
        input_df = pd.concat((input_dates, input_data, quality[-len(input_dates):].reset_index(drop=True)), axis=1)

        # Scaled input data sheet
        scaled_input_df = pd.concat((input_dates, scaled_input_data), axis=1)

        # Scaled output data sheet
        scaled_output_df = pd.concat((output_dates, scaled_output_data), axis=1)

        # Output data sheet
        output_df = pd.concat((output_dates, output_data, output_current_swl, output_swl, quality[-len(output_dates):].reset_index(drop=True)), axis=1)

        # ACF values sheet
        for i in range(len(input_variables)):
            if i == 0:
                acf_df = pd.DataFrame({input_variables[i]: acf(input_data[input_variables[i]], nlags=max(min(40, int(len(self.model_parameters.input_data) / 2 - 1)), self.model_parameters.periods_in))})
            else:
                temp_df = pd.DataFrame({input_variables[i]: acf(input_data[input_variables[i]], nlags=max(min(40, int(len(self.model_parameters.input_data) / 2 - 1)), self.model_parameters.periods_in))})
                acf_df = pd.concat((acf_df, temp_df), axis=1)

        # PACF values sheet
        for i in range(len(input_variables)):
            if i == 0:
                pacf_df = pd.DataFrame({input_variables[i]: pacf(input_data[input_variables[i]], nlags=max(min(40, int(len(self.model_parameters.input_data) / 2 - 1)), self.model_parameters.periods_in))})
            else:
                temp_df = pd.DataFrame({input_variables[i]: pacf(input_data[input_variables[i]], nlags=max(min(40, int(len(self.model_parameters.input_data) / 2 - 1)), self.model_parameters.periods_in))})
                pacf_df = pd.concat((pacf_df, temp_df), axis=1)

        # Correlation value sheet
        corr_df = input_data.corr()

        # LSTM predictions sheets
        train_prediction = self.keras_model.y_hat_train.reshape(-1)
        test_prediction = self.keras_model.y_hat_test.reshape(-1)
        unscaled_train_prediction = self.keras_model.unscaled_y_hat_train.reshape(-1)
        unscaled_test_prediction = self.keras_model.unscaled_y_hat_test.reshape(-1)

        if self.model_parameters.gwl_output in ['delta', 'average_delta']:
            train_level_prediction = self.keras_model.y_hat_level_train.reshape(-1)
            test_level_prediction = self.keras_model.y_hat_level_test.reshape(-1)

            lstm_train_df = pd.DataFrame({'scaled_train_lstm': train_prediction,
                                          'unscaled_train_lstm': unscaled_train_prediction,
                                          'train_level_lstm': train_level_prediction})
            lstm_train_df = pd.concat((train_dates, lstm_train_df), axis=1)

            lstm_test_df = pd.DataFrame({'scaled_test_lstm': test_prediction,
                                         'unscaled_test_lstm': unscaled_test_prediction,
                                         'test_level_lstm': test_level_prediction})
            lstm_test_df = pd.concat((test_dates, lstm_test_df), axis=1)
        else:
            lstm_train_df = pd.DataFrame({'scaled_train_lstm': train_prediction, 'unscaled_train_lstm': unscaled_train_prediction})
            lstm_train_df = pd.concat((train_dates, lstm_train_df), axis=1)
            lstm_test_df = pd.DataFrame({'scaled_test_lstm': test_prediction, 'unscaled_test_lstm': unscaled_test_prediction})
            lstm_test_df = pd.concat((test_dates, lstm_test_df), axis=1)

        # SVR predictions sheets
        train_prediction = self.sklearn_model.y_hat_train.reshape(-1)
        test_prediction = self.sklearn_model.y_hat_test.reshape(-1)
        unscaled_train_prediction = self.sklearn_model.unscaled_y_hat_train.reshape(-1)
        unscaled_test_prediction = self.sklearn_model.unscaled_y_hat_test.reshape(-1)

        if self.model_parameters.gwl_output in ['delta', 'average_delta']:
            train_level_prediction = self.sklearn_model.y_hat_level_train.reshape(-1)
            test_level_prediction = self.sklearn_model.y_hat_level_test.reshape(-1)

            svr_train_df = pd.DataFrame({'scaled_train_svr': train_prediction,
                                         'unscaled_train_svr': unscaled_train_prediction,
                                         'train_level_svr': train_level_prediction})
            svr_train_df = pd.concat((train_dates, svr_train_df), axis=1)

            svr_test_df = pd.DataFrame({'scaled_test_svr': test_prediction,
                                        'unscaled_test_svr': unscaled_test_prediction,
                                        'test_level_svr': test_level_prediction})
            svr_test_df = pd.concat((test_dates, svr_test_df), axis=1)
        else:
            svr_train_df = pd.DataFrame({'scaled_train_svr': train_prediction, 'unscaled_train_svr': unscaled_train_prediction})
            svr_train_df = pd.concat((train_dates, svr_train_df), axis=1)
            svr_test_df = pd.DataFrame({'scaled_test_svr': test_prediction, 'unscaled_test_svr': unscaled_test_prediction})
            svr_test_df = pd.concat((test_dates, svr_test_df), axis=1)

        # LSTM scores sheet
        lstm_scores_df = pd.DataFrame({'score': list(map(lambda x: x[6:], self.keras_model.train_scores_dict.keys())),
                                       'train': list(self.keras_model.train_scores_dict.values()),
                                       'test': list(self.keras_model.test_scores_dict.values())})

        # SVR scores sheet
        svr_scores_df = pd.DataFrame({'score': list(map(lambda x: x[6:], self.sklearn_model.train_scores_dict.keys())),
                                      'train': list(self.sklearn_model.train_scores_dict.values()),
                                      'test': list(self.sklearn_model.test_scores_dict.values())})

        # LSTM history sheet
        training_loss = self.keras_model.history.history['loss']
        epochs = [i + 1 for i in range(len(training_loss))]
        if self.keras_model.val_split > 0:
            validation_loss = self.keras_model.history.history['val_loss']
            lstm_history_df = pd.DataFrame({'epoch': epochs, 'training_loss': training_loss, 'validation_loss': validation_loss})
        else:
            lstm_history_df = pd.DataFrame({'epoch': epochs, 'training_loss': training_loss})

        # SVR history sheet
        epoch = [i + 1 for i in range(len(self.sklearn_model.epoch_list))]
        num_examples = self.sklearn_model.epoch_list
        svr_history_df = pd.DataFrame({'epoch': epoch, 'num_exmaples': num_examples})
        train_scores_df = pd.DataFrame(self.sklearn_model.train_curve_scores)
        train_scores_df.columns = [f'train fold {i+1}' for i in range(len(train_scores_df.columns))]
        val_scores_df = pd.DataFrame(self.sklearn_model.test_curve_scores)
        val_scores_df.columns = [f'validation fold {i+1}' for i in range(len(val_scores_df.columns))]
        fit_times_df = pd.DataFrame(self.sklearn_model.fit_times)
        fit_times_df.columns = [f'fit_times fold {i+1}' for i in range(len(fit_times_df.columns))]
        score_times_df = pd.DataFrame(self.sklearn_model.score_times)
        score_times_df.columns = [f'score_times fold {i+1}' for i in range(len(score_times_df.columns))]
        svr_history_df = pd.concat((svr_history_df, train_scores_df, val_scores_df, fit_times_df, score_times_df), axis=1)

        with pd.ExcelWriter(spreadsheet_file_path) as writer:
            general_info_df.to_excel(writer, sheet_name='general_info')
            input_df.to_excel(writer, sheet_name='input_data')
            scaled_input_df.to_excel(writer, sheet_name='scaled_input_data')
            output_df.to_excel(writer, sheet_name='output_data')
            scaled_output_df.to_excel(writer, sheet_name='scaled_output_data')
            acf_df.to_excel(writer, sheet_name='acf_values')
            pacf_df.to_excel(writer, sheet_name='pacf_values')
            corr_df.to_excel(writer, sheet_name='correlation_values')
            lstm_train_df.to_excel(writer, sheet_name='lstm_train_predictions')
            lstm_test_df.to_excel(writer, sheet_name='lstm_test_predictions')
            lstm_scores_df.to_excel(writer, sheet_name='lstm_scores')
            svr_train_df.to_excel(writer, sheet_name='svr_train_predictions')
            svr_test_df.to_excel(writer, sheet_name='svr_test_predictions')
            svr_scores_df.to_excel(writer, sheet_name='svr_scores')
            lstm_history_df.to_excel(writer, sheet_name='lstm_history')
            svr_history_df.to_excel(writer, sheet_name='svr_history')


    def save_models(self):
        file_name_date_time = self.start_date_time.strftime("%Y_%m_%d__%H_%M_%S")

        # Save LSTM Model
        lstm_model_file_name = f'lstm_model_{file_name_date_time}'
        lstm_model_file_path = import_functions.make_path(f'{self.bore_id}/models', lstm_model_file_name)
        self.keras_model.model.save(lstm_model_file_path)

        # Save SVR Model
        svr_model_file_name = f'svr_model_{file_name_date_time}.pkl'
        svr_model_file_path = import_functions.make_path(f'{self.bore_id}/models', svr_model_file_name)
        pickle.dump(self.sklearn_model.model, open(svr_model_file_path, 'wb'))

        # Load Models
        # lstm_model = tf.keras.models.load_model(model_file_path)
        # svr_model = pickle.load(open(svr_model_file_path, 'rb'))
