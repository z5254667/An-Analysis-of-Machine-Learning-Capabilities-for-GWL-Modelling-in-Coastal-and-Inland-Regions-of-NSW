#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:41:23 2022

@author: johnsalvaris
"""

# Import packages
# import glob
import os
import numbers
import datetime as dt
import netCDF4 as nc
import numpy as np
import pandas as pd

from pathlib import Path

def clear_temp_folder(bore_id):
    # temp_files = glob.glob(f'{os.getcwd()}/{bore_id}/temp/*', recursive=True)
    temp_files = Path(f'{os.getcwd()}/{bore_id}/temp/').glob('*')
    for t in temp_files:
        os.remove(t)


def make_path(subfolders, file_name):
    main_folder = Path(os.getcwd())  # '/Users/johnsalvaris/OneDrive - UNSW/Thesis-John’s MacBook Pro 2020/'

    return main_folder / subfolders / file_name


def get_bore_df(bore_id):
    # subfolders = f'/{bore_id}/gwe_data/'
    subfolders = f'{bore_id}/gwe_data'
    bore_file = 'Bore Detailed.csv'
    bore_path = make_path(subfolders, bore_file)
    bore_df = pd.read_csv(bore_path)

    return bore_df


def get_gwl_df(bore_id):
    # subfolders = f'/{bore_id}/gwe_data/'
    subfolders = f'{bore_id}/gwe_data'
    gwl_file = 'Water Level.csv'
    gwl_path = make_path(subfolders, gwl_file)

    gwl_df = pd.read_csv(gwl_path, parse_dates=['Date'])
    gwl_df = gwl_df[gwl_df['Variable'] == 'SWL']  # Only use standing water level measurments
    gwl_df = gwl_df.sort_values(by='Date', ascending=True)  # Force to be ascending date order
    gwl_df = gwl_df.reset_index(drop=True)

    return gwl_df


def get_silo_df(bore_id):
    # subfolders = f'/{bore_id}/silo_data/'
    subfolders = f'{bore_id}/silo_data'
    silo_file = 'SILO.csv'
    silo_path = make_path(subfolders, silo_file)

    silo_df = pd.read_csv(silo_path, parse_dates=['YYYY-MM-DD'])
    silo_df = silo_df.sort_values(by='YYYY-MM-DD', ascending=True)  # Force to be ascending date order

    return silo_df


def get_awo_df(bore_id, latitude, longitude):
    # subfolders = f'/{bore_id}/awo_data/'
    subfolders = f'{bore_id}/awo_data'

    if bore_id == 'GW036872.1.1' or bore_id == 'GW081101.1.2':
        first_year = 2010
    elif bore_id == 'GW075025.1.1':
        first_year = 2000
    elif bore_id == 'GW075405.1.1':
        first_year = 2012
    elif bore_id == 'GW080079.1.1':
        first_year = 2002
    elif bore_id == 'GW080415.1.1':
        first_year = 2005
    elif bore_id == 'GW080980.1.1':
        first_year = 2006
    elif bore_id == 'GW273314.1.1':
        first_year = 2014
    elif bore_id == 'GW403746.3.3':
        first_year = 2009
    else:
        raise Exception('Invaild bore id selected.')

    for year in range(first_year, 2023):
        # Absolute Total Rootzone Soil Moisture (0-100cm) (equal to sum of upper and lower layers)
        topsoil_file = f'sm_pct_{year}.nc'
        topsoil_path = make_path(subfolders, topsoil_file)
        topsoil_ds = nc.Dataset(topsoil_path)
        long_index = np.where(topsoil_ds['longitude'][:] == longitude)[0][0]
        lat_index = np.where(topsoil_ds['latitude'][:] == latitude)[0][0]
        topsoil_data = topsoil_ds['sm_pct'][:, lat_index, long_index]
        dates = dt.date(1900, 1, 1) + np.array(list(map(lambda x: dt.timedelta(int(x)), topsoil_ds['time'][:][:])))  # Applies to all nc files
        dates = pd.to_datetime(dates)

        # Absolute Total Upper Layer Soil Moisture (0-10cm)
        upper_layer_file = f's0_pct_{year}.nc'
        upper_layer_path = make_path(subfolders, upper_layer_file)
        upper_layer_ds = nc.Dataset(upper_layer_path)
        long_index = np.where(upper_layer_ds['longitude'][:] == longitude)[0][0]
        lat_index = np.where(upper_layer_ds['latitude'][:] == latitude)[0][0]
        upper_data = upper_layer_ds['s0_pct'][:, lat_index, long_index]

        # Absolute Total Lower Layer Soil Moisture (10-100cm)
        lower_layer_file = f'ss_pct_{year}.nc'
        lower_layer_path = make_path(subfolders, lower_layer_file)
        lower_layer_ds = nc.Dataset(lower_layer_path)
        long_index = np.where(lower_layer_ds['longitude'][:] == longitude)[0][0]
        lat_index = np.where(lower_layer_ds['latitude'][:] == latitude)[0][0]
        lower_data = lower_layer_ds['ss_pct'][:, lat_index, long_index]

        # Absolute Total Deep Layer Soil Moisture (1-6m)
        deep_layer_file = f'sd_pct_{year}.nc'
        deep_layer_path = make_path(subfolders, deep_layer_file)
        deep_layer_ds = nc.Dataset(deep_layer_path)
        long_index = np.where(deep_layer_ds['longitude'][:] == longitude)[0][0]
        lat_index = np.where(deep_layer_ds['latitude'][:] == latitude)[0][0]
        deep_data = deep_layer_ds['sd_pct'][:, lat_index, long_index]

        # Absolute Deep Drainage (below 6m)
        drainage_file = f'dd_{year}.nc'
        drainage_path = make_path(subfolders, drainage_file)
        drainage_ds = nc.Dataset(drainage_path)
        long_index = np.where(drainage_ds['longitude'][:] == longitude)[0][0]
        lat_index = np.where(drainage_ds['latitude'][:] == latitude)[0][0]
        drainage_data = drainage_ds['dd'][:, lat_index, long_index]

        if year == first_year:
            awo_df = pd.DataFrame({'date': dates,
                                   'sm_pct': topsoil_data,
                                   's0_pct': upper_data,
                                   'ss_pct': lower_data,
                                   'sd_pct': deep_data,
                                   'dd': drainage_data})
        else:
            temp_awo_df = pd.DataFrame({'date': dates,
                                        'sm_pct': topsoil_data,
                                        's0_pct': upper_data,
                                        'ss_pct': lower_data,
                                        'sd_pct': deep_data,
                                        'dd': drainage_data})
            awo_df = pd.concat((awo_df, temp_awo_df)).reset_index(drop=True)

    return awo_df


def shorten_dataset(dataset, years_of_data='all', use_newest_data=True, fix_to_calendar=False, first_date=None):
    if fix_to_calendar and not isinstance(first_date, pd._libs.tslibs.timestamps.Timestamp):
        raise Exception("first_date must have the pandas Timestamp data type or not be included.")

    dataset = dataset.to_numpy()  # Dataset provided should be a Pandas dataframe series

    if years_of_data == 'all':
        return dataset
    elif isinstance(years_of_data, numbers.Number):
        year_days = int(years_of_data * 365)

        # Check range resides in the dataset
        if year_days > len(dataset):
            raise Exception(f"The years_of_data specified ({years_of_data} years) is larger than the available dataset. Choose a value for years_of_data smaller than {round(len(dataset) / 365, 4)} years or specify years_of_data = 'all' to use the entire avaiable dataset.")

        if use_newest_data:  # Keep the newest data available
            if fix_to_calendar:  # Shortened dataset will end on 31st December
                last_date = first_date + dt.timedelta(days=len(dataset) - 1)

                if int(dt.datetime.strftime(last_date, "%m")) + int(dt.datetime.strftime(last_date, "%d")) != 43:  # Check data does not end on 31st December already
                    end_of_year = pd.Timestamp(year=int(dt.datetime.strftime(last_date, "%Y")) - 1, month=12, day=31)
                    remove_from_end = (last_date - end_of_year).days
                    # Check range resides in the dataset
                    if year_days > len(dataset[:-remove_from_end]):
                        raise Exception(f"The years_of_data specified ({years_of_data} years) is too large to fix to 31st Decemeber. Choose a value for years_of_data smaller than {round(len(dataset[:-remove_from_end]) / 365, 4)} years or specify fix_to_calendar = False to use the final data point available.")
                    else:
                        return dataset[-remove_from_end - year_days:-remove_from_end]
            return dataset[-year_days:]
        else:  # Keep oldest data available
            if fix_to_calendar:  # Shortened dataset will start on 1st January
                if int(dt.datetime.strftime(first_date, "%m")) + int(dt.datetime.strftime(first_date, "%d")) != 2:  # Check data does not start on 1st January already
                    start_of_year = pd.Timestamp(year=int(dt.datetime.strftime(first_date, "%Y")) + 1, month=1, day=1)
                    remove_from_start = (start_of_year - first_date).days
                    # Check range resides in the dataset
                    if year_days > len(dataset[remove_from_start:]):
                        raise Exception(f"The years_of_data specified ({years_of_data} years) is too large to fix to 1st January. Choose a value for years_of_data smaller than {round(len(dataset[remove_from_start:]) / 365, 4)} years or specify fix_to_calendar = False to use the first data point available.")
                    else:
                        return dataset[remove_from_start: year_days + remove_from_start]
            return dataset[:year_days]
    else:
        raise Exception("The years_of_data parameter must be 'all' or a number")
