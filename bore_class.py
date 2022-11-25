#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:43:55 2022

@author: johnsalvaris
"""

# Import created scripts
import import_functions

# Import packages
import numpy as np
import pandas as pd


class bore():
    def __init__(self, bore_id, use_awo=True, years_of_data='all', use_newest_data=True, fix_to_calendar=False):
        self.id = bore_id
        self.use_awo = use_awo
        self.years_of_data = years_of_data
        self.use_newest_data = use_newest_data
        self.fix_to_calendar = fix_to_calendar

        # Classify region
        coastal_ids = ['GW036872.1.1', 'GW075025.1.1', 'GW080079.1.1', 'GW080980.1.1', 'GW081101.1.2']
        inland_ids = ['GW075405.1.1', 'GW080415.1.1', 'GW273314.1.1', 'GW403746.3.3']

        if self.id in coastal_ids:
            self.region = 'Coastal'
        elif self.id in inland_ids:
            self.region = 'Inland'
        else:
            self.region = 'Unknown'


    def add_dfs(self):
        # Add store dataframes to bore
        self.bore_df = import_functions.get_bore_df(self.id)
        self.gwl_df = import_functions.get_gwl_df(self.id)
        self.silo_df = import_functions.get_silo_df(self.id)

        # Add locations (allows awo_df to be added)
        self.bore_latitude = self.bore_df['Latitude'][0]
        self.bore_longitude = self.bore_df['Longitude'][0]
        self.silo_latitude = self.silo_df['latitude'].to_numpy()[0]
        self.silo_longitude = self.silo_df['longitude'].to_numpy()[0]

        # Include awo data
        if self.use_awo:
            self.awo_latitude = self.silo_latitude  # same as SILO grid point for data consistency
            self.awo_longitude = self.silo_longitude  # same as SILO grid point for data consistency

            # Add awo_df to bore
            self.awo_df = import_functions.get_awo_df(self.id, self.awo_latitude, self.awo_longitude)


    def handle_missing_dates(self, interpolation_method='Spline'):
        # Remove sequences from start/end with missing day gaps if will not shrink the dataset by more than ~5%
        if self.id == 'GW075025.1.1':
            self.gwl_df = self.gwl_df.drop(index=self.gwl_df.index[:305]).reset_index(drop=True)  # Slicing chosen manually
        elif self.id == 'GW075405.1.1':
            self.gwl_df = self.gwl_df.drop(index=self.gwl_df.index[:168]).reset_index(drop=True)  # Slicing chosen manually
        elif self.id == 'GW080415.1.1':
            self.gwl_df = self.gwl_df.drop(index=self.gwl_df.index[:223]).reset_index(drop=True)  # Slicing chosen manually
        elif self.id == 'GW080980.1.1':
            self.gwl_df = self.gwl_df.drop(index=self.gwl_df.index[:2]).reset_index(drop=True)  # Slicing chosen manually
            self.gwl_df = self.gwl_df.drop(index=self.gwl_df.index[-214:]).reset_index(drop=True)  # Slicing chosen manually
        elif self.id == 'GW403746.3.3':
            self.gwl_df = self.gwl_df.drop(index=self.gwl_df.index[:69]).reset_index(drop=True)  # Slicing chosen manually
        elif self.id == 'GW080079.1.1':
            # Bore to be deleted
            pass

        # Interpolate any small gaps remaining in the dataset
        self.interpolation_method = interpolation_method
        temp_gwl_dates = self.gwl_df['Date'].to_numpy()  # Array of the dates in gwl_df
        dayskip = np.array([np.timedelta64(temp_gwl_dates[i] - temp_gwl_dates[i - 1], 'D').astype(int) for i in range(1, len(temp_gwl_dates))])  # days between each row
        dates_index = np.array([i for i in range(len(temp_gwl_dates))])
        missing_days = dayskip - 1  # Missing days betweene each row
        insert_fraction = 1 / dayskip  # Fraction for each dayskip so consecutive missing days can be inserted correctly
        start_index = dates_index[1:] - 1 + insert_fraction
        end_index = dates_index[1:] - insert_fraction
        insert_indexs = np.array([])
        for i in range(len(missing_days)):  # All locations where new rows will be inserted
            insert_indexs = np.append(insert_indexs, np.linspace(start_index[i], end_index[i], missing_days[i]).reshape(-1))
        # Place holder dataframe with generic information to be added to the main dataframe, indexed by the locations where new rows are required
        fresh_lines = pd.DataFrame({'Bore ID': self.id, 'Variable': 'SWL', 'Quality Code': 'quality-I', 'Agency': f'Interpolation - {self.interpolation_method}'}, index=insert_indexs)
        # Combine place holder dataframe with main dataframe
        self.gwl_df = pd.concat((self.gwl_df, fresh_lines), ignore_index=False)
        self.gwl_df = self.gwl_df.sort_index().reset_index(drop=True)
        # Interpolate missing swl values
        self.gwl_df['Result (m)'] = self.gwl_df['Result (m)'].interpolate(option=self.interpolation_method.lower())  # IMPORTANT NOTE option= IS NOT CORRECT --> MUST BE method= , order= --> Therefore all Linearly interpolated
        # Add corresponding dates to interpolated swl values
        for i in range(len(self.gwl_df)):
            if pd.isnull(self.gwl_df['Date'][i]):
                self.gwl_df.loc[i, 'Date'] = self.gwl_df['Date'][i - 1] + np.timedelta64(1, 'D')


    def remove_null_dates(self):
        gwl_dates = self.gwl_df['Date'].to_numpy()
        self.silo_df = self.silo_df[self.silo_df['YYYY-MM-DD'].isin(gwl_dates)]  # remove dates from silo data not in gwl data
        self.silo_df = self.silo_df.reset_index(drop=True)
        if self.use_awo:
            self.awo_df = self.awo_df[self.awo_df['date'].isin(gwl_dates)]  # remove dates from awo data not in gwl data
            self.awo_df = self.awo_df.reset_index(drop=True)


    def add_silo_data(self):
        # Adds all variable data to the bore class as arrays
        silo_dates = self.silo_df['YYYY-MM-DD']

        self.daily_rain = import_functions.shorten_dataset(self.silo_df['daily_rain'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.max_temp = import_functions.shorten_dataset(self.silo_df['max_temp'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.min_temp = import_functions.shorten_dataset(self.silo_df['min_temp'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.vp = import_functions.shorten_dataset(self.silo_df['vp'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.vp_deficit = import_functions.shorten_dataset(self.silo_df['vp_deficit'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.evap_pan = import_functions.shorten_dataset(self.silo_df['evap_pan'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.evap_syn = import_functions.shorten_dataset(self.silo_df['evap_syn'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.evap_comb = import_functions.shorten_dataset(self.silo_df['evap_comb'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.evap_morton_lake = import_functions.shorten_dataset(self.silo_df['evap_morton_lake'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.radiation = import_functions.shorten_dataset(self.silo_df['radiation'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.rh_tmax = import_functions.shorten_dataset(self.silo_df['rh_tmax'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.rh_tmin = import_functions.shorten_dataset(self.silo_df['rh_tmin'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.et_short_crop = import_functions.shorten_dataset(self.silo_df['et_short_crop'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.et_tall_crop = import_functions.shorten_dataset(self.silo_df['et_tall_crop'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.et_morton_actual = import_functions.shorten_dataset(self.silo_df['et_morton_actual'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.et_morton_potential = import_functions.shorten_dataset(self.silo_df['et_morton_potential'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.et_morton_wet = import_functions.shorten_dataset(self.silo_df['et_morton_wet'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])
        self.mslp = import_functions.shorten_dataset(self.silo_df['mslp'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, silo_dates[0])


    def add_awo_data(self):
        # Adds all variable data to the bore class as arrays
        awo_dates = self.awo_df['date']

        self.sm_pct = import_functions.shorten_dataset(self.awo_df['sm_pct'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, awo_dates[0])
        self.s0_pct = import_functions.shorten_dataset(self.awo_df['s0_pct'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, awo_dates[0])
        self.ss_pct = import_functions.shorten_dataset(self.awo_df['ss_pct'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, awo_dates[0])
        self.sd_pct = import_functions.shorten_dataset(self.awo_df['sd_pct'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, awo_dates[0])
        self.dd = import_functions.shorten_dataset(self.awo_df['dd'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, awo_dates[0])


    def add_gwl_data(self, leads):
        self.leads = leads
        # Addas all variable data to the bore class as arrays
        gwl_dates = self.gwl_df['Date']
        self.swl = import_functions.shorten_dataset(self.gwl_df['Result (m)'], self.years_of_data, self.use_newest_data, self.fix_to_calendar, gwl_dates[0])
        self.dates = import_functions.shorten_dataset(gwl_dates, self.years_of_data, self.use_newest_data, self.fix_to_calendar, gwl_dates[0])

        # Calculate DAILY change in swl for an output
        self.output_delta_swl = []
        for i in range(self.leads, len(self.swl)):
            self.output_delta_swl.append(self.swl[i] - self.swl[i - self.leads])
        self.output_delta_swl = np.array(self.output_delta_swl)  # length = length of swl - number of leads

        # Calculate DAILY change in swl for an input
        self.daily_delta_swl = []
        for i in range(1, len(self.swl)):
            self.daily_delta_swl.append(self.swl[i] - self.swl[i - 1])
        self.daily_delta_swl = np.array(self.daily_delta_swl)  # length = length of swl - 1


    def average_data(self, av_period=30):
        self.av_period = av_period

        def convert_to_period_avs(data, time_period):
            # data must be one dimensional
            num_periods = len(data) // time_period
            groups = np.array([])
            for i in range(num_periods):
                groups = np.append(groups, data[i * time_period:time_period * (i + 1)])
            groups = groups.reshape((num_periods, time_period))
            group_avs = np.array([])
            for group in groups:
                group_avs = np.append(group_avs, np.mean(group))
            return group_avs

        self.av_swl = convert_to_period_avs(self.swl, self.av_period)
        self.av_daily_rain = convert_to_period_avs(self.daily_rain, self.av_period)
        self.av_max_temp = convert_to_period_avs(self.max_temp, self.av_period)
        self.av_min_temp = convert_to_period_avs(self.min_temp, self.av_period)
        self.av_vp = convert_to_period_avs(self.vp, self.av_period)
        self.av_vp_deficit = convert_to_period_avs(self.vp_deficit, self.av_period)
        self.av_evap_pan = convert_to_period_avs(self.evap_pan, self.av_period)
        self.av_evap_syn = convert_to_period_avs(self.evap_syn, self.av_period)
        self.av_evap_comb = convert_to_period_avs(self.evap_comb, self.av_period)
        self.av_evap_morton_lake = convert_to_period_avs(self.evap_morton_lake, self.av_period)
        self.av_radiation = convert_to_period_avs(self.radiation, self.av_period)
        self.av_rh_tmax = convert_to_period_avs(self.rh_tmax, self.av_period)
        self.av_rh_tmin = convert_to_period_avs(self.rh_tmin, self.av_period)
        self.av_et_short_crop = convert_to_period_avs(self.et_short_crop, self.av_period)
        self.av_et_tall_crop = convert_to_period_avs(self.et_tall_crop, self.av_period)
        self.av_et_morton_actual = convert_to_period_avs(self.et_morton_actual, self.av_period)
        self.av_et_morton_potential = convert_to_period_avs(self.et_morton_potential, self.av_period)
        self.av_et_morton_wet = convert_to_period_avs(self.et_morton_wet, self.av_period)
        self.av_mslp = convert_to_period_avs(self.mslp, self.av_period)
        if self.use_awo:
            self.av_sm_pct = convert_to_period_avs(self.sm_pct, self.av_period)
            self.av_s0_pct = convert_to_period_avs(self.s0_pct, self.av_period)
            self.av_ss_pct = convert_to_period_avs(self.ss_pct, self.av_period)
            self.av_sd_pct = convert_to_period_avs(self.sd_pct, self.av_period)
            self.av_dd = convert_to_period_avs(self.dd, self.av_period)

        # calculate change in average swl (change calculated after average - not average calculated after change) --> FOR OUTPUTS
        self.av_output_delta_swl = np.array([])
        for i in range(self.leads, len(self.av_swl)):
            self.av_output_delta_swl = np.append(self.av_output_delta_swl, self.av_swl[i] - self.av_swl[i - self.leads])  # length is = length of av_swl - number of leads

        # calculate average period delta swl (change calculated after average - not average calculated after change) --> FOR INPUTS
        self.av_period_delta_swl = np.array([])
        for i in range(1, len(self.av_swl)):
            self.av_period_delta_swl = np.append(self.av_period_delta_swl, self.av_swl[i] - self.av_swl[i - 1])

        # self.av_daily_delta_swl = convert_to_period_avs(self.daily_delta_swl, self.av_period)
        # self.av_output_delta_swl = convert_to_period_avs(self.output_delta_swl, self.av_period)

        self.av_dates = np.array([self.dates[i] for i in range(len(self.dates)) if i % self.av_period == self.av_period - 1])  # Average date taken as the last date in the period being averaged


    def add_data_dict(self):
        self.data_dict = {'swl': self.swl,
                          'daily_rain': self.daily_rain,
                          'max_temp': self.max_temp,
                          'min_temp': self.min_temp,
                          'vp': self.vp,
                          'vp_deficit': self.vp_deficit,
                          'evap_pan': self.evap_pan,
                          'evap_syn': self.evap_syn,
                          'evap_comb': self.evap_comb,
                          'evap_morton_lake': self.evap_morton_lake,
                          'radiation': self.radiation,
                          'rh_tmax': self.rh_tmax,
                          'rh_tmin': self.rh_tmin,
                          'et_short_crop': self.et_short_crop,
                          'et_tall_crop': self.et_tall_crop,
                          'et_morton_actual': self.et_morton_actual,
                          'et_morton_potential': self.et_morton_potential,
                          'et_morton_wet': self.et_morton_wet,
                          'mslp': self.mslp,
                          'daily_delta_swl': self.daily_delta_swl,
                          'delta_swl': self.output_delta_swl,
                          'av_swl': self.av_swl,
                          'av_daily_rain': self.av_daily_rain,
                          'av_max_temp': self.av_max_temp,
                          'av_min_temp': self.av_min_temp,
                          'av_vp': self.av_vp,
                          'av_vp_deficit': self.av_vp_deficit,
                          'av_evap_pan': self.av_evap_pan,
                          'av_evap_syn': self.av_evap_syn,
                          'av_evap_comb': self.av_evap_comb,
                          'av_evap_morton_lake': self.av_evap_morton_lake,
                          'av_radiation': self.av_radiation,
                          'av_rh_tmax': self.av_rh_tmax,
                          'av_rh_tmin': self.av_rh_tmin,
                          'av_et_short_crop': self.av_et_short_crop,
                          'av_et_tall_crop': self.av_et_tall_crop,
                          'av_et_morton_actual': self.av_et_morton_actual,
                          'av_et_morton_potential': self.av_et_morton_potential,
                          'av_et_morton_wet': self.av_et_morton_wet,
                          'av_mslp': self.av_mslp,
                          'av_period_delta_swl': self.av_period_delta_swl,
                          'av_delta_swl': self.av_output_delta_swl}

        if self.use_awo:
            self.data_dict['sm_pct'] = self.sm_pct
            self.data_dict['s0_pct'] = self.s0_pct
            self.data_dict['ss_pct'] = self.ss_pct
            self.data_dict['sd_pct'] = self.sd_pct
            self.data_dict['dd'] = self.dd
            self.data_dict['av_sm_pct'] = self.av_sm_pct
            self.data_dict['av_s0_pct'] = self.av_s0_pct
            self.data_dict['av_ss_pct'] = self.av_ss_pct
            self.data_dict['av_sd_pct'] = self.av_sd_pct
            self.data_dict['av_dd'] = self.av_dd
