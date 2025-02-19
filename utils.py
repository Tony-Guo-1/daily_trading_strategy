import pandas as pd
import numpy as np
import math

# For feature engineering
from ta import add_all_ta_features

# For CNN class
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Embedding, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout, Dense, Add
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, AveragePooling1D, MaxPool1D
from keras import backend as K

## Data Processing Functions
def remove_tic(data, tickers, num_of_ts, factors):
    '''
    Removes tickers without enough dates
    '''

    # Remove entries where there are not enough number of time stamps
    databytic_count = data.groupby('tic').count()
    incomplete_tics = databytic_count.loc[databytic_count['datadate'] < num_of_ts].index.tolist()
    print(f'Removed {incomplete_tics}')
    data = data[~data['tic'].isin(incomplete_tics)]
    tickers = data['tic'].unique()
    print(f'There are {len(incomplete_tics)} unique tickers to remove due to insufficient dates')
    print(f'There are {len(tickers)} tickers')

    print(f'Confirm there are no more columns with missing data {data.isna().any().any()}')
    
    # Sort data by tickers and date
    data = data.sort_values(by=['tic','datadate'])
    data = data.reset_index(drop=True)
    
    return data, tickers

def remove_low_dollar_vol(data, tickers, dol_vol_thres = 10000000):
    '''
    Removes tickers whose dollar volume is less than dol_vol_thres (default 10M)
    '''
    data['dol_vol'] = data['cshtrd'] * data['prcod']
    dolvol_means = data.groupby('tic')['dol_vol'].mean()
    dolvol_remove = dolvol_means[dolvol_means < dol_vol_thres].index
    print(f'Removed {len(dolvol_remove)} tickers due to low dollar volume')
    data = data[~data['tic'].isin(dolvol_remove)]
    tickers = data['tic'].unique()
    print('There are ' + str(len(tickers)) + ' tickers')
    return data, tickers

def compute_ret(data, TBill_file_path):
    '''
    1. Creates ret_d colume, e.g. 2024-2-2's ret_d is the percentage return of buying at 2024-2-3 market open and selling at 2024-2-4 market open
    2. Fills last two days of missing return as 0
    3. Creates the column of risk-free rate from the 1y TBill rate for SR calculation later
    '''
    # Calculate the daily return
    data['ret_d'] = data.groupby('tic')['prcod'].pct_change()
    data['ret_d'] = data.groupby('tic')['ret_d'].shift(-2)
    
    # Currently the last two days have no ret_d since it's the test date. Fill it with 0
    data = data.fillna(0)
    data = data.reset_index(drop=True)
    
    # Excess return
    TBill1y = pd.read_csv(TBill_file_path)
    TBill1y = TBill1y.replace('.', None)
    TBill1y = TBill1y.ffill()
    TBill1y['TBill_rate'] = TBill1y['TBill_rate'].apply(lambda x: float(x))
    TBill1y['date'] = pd.to_datetime(TBill1y['date'])
    date_TBill_dict = TBill1y[['date', 'TBill_rate']].set_index('date')['TBill_rate'].to_dict()
    data['TBill1y'] = data['datadate'].map(date_TBill_dict)
    # Convert to daily rate
    data['TBill1y'] = data['TBill1y'].apply(lambda x: np.power(1 + x/100, 1/252) - 1)
    
    # Relative return to equal-weighted market returns
    data['market_ret'] = data['datadate'].map(data.groupby('datadate')['ret_d'].mean())
    data['rel_ret_d'] = data['ret_d'] - data['market_ret']
    data = data.drop(columns=['market_ret'])
    
    return data

def remove_dead_stocks(data, price_thres = 0.1):
    '''
    Changes dead stock's return and their rank to 0 five days before their price reaches the threshold
    '''
    df_low_price = data[data['prcod'] <= price_thres]
    tic_low_price = list(df_low_price.tic.unique())
    all_days = list(data.datadate.unique())
    for tic in tic_low_price:
        date_low_price = df_low_price[df_low_price['tic']==tic].datadate.values[0]
        date_index = all_days.index(date_low_price)
        data.loc[(data['tic'] == tic) & (data['datadate'] >= all_days[date_index-5]), ['ret_d', 'rel_ret_d', 'rank']] = 0
        print(f'{tic} is dead after {date_low_price}')
    return data

def fixed_thres_classes(x):
    '''
    Transforms daily return to one of 5 labels
    '''
    if x <= -0.03:
        return -2
    elif x <= -0.01:
        return -1
    elif x <= 0.01:
        return 0
    elif x < 0.03:
        return 1
    else:
        return 2
    
def assign_class_labels(data, label_method):
    '''
    Computes the daily rank (-2, -1, 0, 1, 2) of every stock using "fixed_thres_classes"
    '''
    # Manually set the bin threshold to be \pm 0.01 and \pm 0.03
    if label_method == 'fixed_thres':
        data['rank'] = data['ret_d'].apply(fixed_thres_classes)
    else:
        raise Exception('Nonexistent label method!')
    return data

def make_sector_column(data):
    '''
    Creates the categorical sector column, named 0, 1, ...
    '''

    num_of_tokens = data.sector.nunique()
    num_to_sector_dict = {}
    
    # Renumber the sics from 0 to num_of_sectors-1
    all_sectors = list(data.sector.unique())
    renumber_sectors = {}
    for i in range(len(all_sectors)):
        renumber_sectors[all_sectors[i]] = i
        num_to_sector_dict[i] = all_sectors[i]
    data['sector'] = data['sector'].replace(renumber_sectors)
    
    return data, num_of_tokens, num_to_sector_dict

def num_tic_dicts(data):
    '''
    Creates two dictionaries corresponding a numerical label to tic symbol and vice versa for each stock
    '''
    all_tickers = list(data['tic'].unique())
    all_tickers.sort()
    l = len(all_tickers)
    
    num_to_tic_dict = {}
    for i in range(l):
        num_to_tic_dict[i] = all_tickers[i]
    
    tic_to_num_dict = {}
    for key, value in num_to_tic_dict.items():
        tic_to_num_dict[value] = key
    
    return num_to_tic_dict, tic_to_num_dict

## Feature Engineering Functions
# Basic Features
def momentum_235(data, factors):
    '''
    Calculates 2,3,5-day (percentage) momentums of opening prices
    '''
    data['Mom_2day'] = data.groupby('tic')['prcod'].pct_change(periods=2)
    factors.append('Mom_2day')
    data['Mom_3day'] = data.groupby('tic')['prcod'].pct_change(periods=3)
    factors.append('Mom_3day')
    data['Mom_5day'] = data.groupby('tic')['prcod'].pct_change(periods=5)
    factors.append('Mom_5day')
    return data, factors

def MA_1050(data, factors):
    '''
    Calculates 10,50-day simple moving averages of opening prices
    '''
    data['MA_10day'] = data.groupby('tic')['prcod'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    data['MA_50day'] = data.groupby('tic')['prcod'].transform(lambda x: x.rolling(50, min_periods=1).mean())
    factors.append('MA_10day')
    factors.append('MA_50day')
    return data, factors

def price_vs_MA(data, factors):
    '''
    Calculates the ratio of opening prices and 10,50-day moving averages
    '''
    data['open/MA10'] = data['prcod'] / data['MA_10day']
    data['open/MA50'] = data['prcod'] / data['MA_50day']
    factors.append('open/MA10')
    factors.append('open/MA50')
    return data, factors

def STD_10(data, factors):
    '''
    Calculates the 10-day moving standard deviation of opening price
    '''
    data['STD_10day'] = data.groupby('tic')['prcod'].transform(lambda x: x.rolling(10, min_periods=1).std())
    factors.append('STD_10day')  
    return data, factors

def H_L(data, factors):
    '''
    Calculates the daily spread: high - low
    '''
    data['H-L'] = data['prchd'] - data['prcld']
    factors.append('H-L')
    return data, factors

def RSI_14(data, factors):
    '''
    Calculates the relative strength index (RSI) using 14-day period
    '''
    data['delta'] = data.groupby('tic')['prcod'].diff()
    data['gain'] = data['delta'].clip(lower=0)
    data['loss'] = -data['delta'].clip(upper=0)
    data['avg_gain'] = data.groupby('tic')['gain'].rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
    data['avg_loss'] = data.groupby('tic')['loss'].rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
    data['RSI'] = 100 - (100 / (1 + data['avg_gain'] / data['avg_loss']))
    data = data.drop(columns=['delta', 'gain', 'loss', 'avg_gain', 'avg_loss'])
    data = data.fillna(0)
    data['RSI'] = data.groupby('tic')['RSI'].transform(lambda x: x.replace(0, x[x != 0].mean()))
    factors.append('RSI')
    return data, factors

def MACD_Line(data, factors):
    '''
    Calculates the Moving Average Convergence Divergence (MACD)
    MACD = EMA12 - EMA26
    MACD_Signal_Line = 9-day exponential moving average of MACD
    '''
    data['EMA12'] = data['prcod'].ewm(span=12, adjust=False, min_periods=1).mean().reset_index(drop=True)
    data['EMA26'] = data['prcod'].ewm(span=26, adjust=False, min_periods=1).mean().reset_index(drop=True)
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal_Line'] = data['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    
    data = data.drop(columns=['EMA12', 'EMA26'])    
    data = data.fillna(0)
    
    factors.extend(['MACD', 'MACD_Signal_Line'])
    
    return data, factors

def all_features_ta(data):
    '''
    Calculates all features in library ta (Technical Analysis Library)
    Note that "opening" and "closing" prices are reversed because all tradings are done at opening prices
    '''
    data = add_all_ta_features(data, open="prccd", high="prchd", low="prcld", close="prcod", volume="cshtrd", fillna=True)
    
    return data

def feature_engineer(data, factors):
    '''
    Computes all features defined above
    '''
    data, factors = momentum_235(data, factors)
    data, factors = MA_1050(data, factors)
    data, factors = price_vs_MA(data, factors)
    data, factors = STD_10(data, factors)
    data, factors = H_L(data, factors)
    data, factors = RSI_14(data, factors)
    data, factors = MACD_Line(data, factors)
    
    print(f'Confirm data has no NAs: {~data.isna().any().any()}')
    
    return data, factors

## Training Functions
def prep_train_test_data(data, seq_length, ftd, ltd, all_days):
    '''
    Prepares the training/validation and test data for CNN
    '''

    num_st_days = 200 # Number of days used for standardization
    if ftd < num_st_days:
        print('not enough standardization days')
    
    first_st_day = all_days[ftd-num_st_days]
    last_st_day = all_days[ftd-1]
    first_train_day = all_days[ftd]
    last_train_day = all_days[ltd]
    first_test_day = all_days[ltd-seq_length+2]
    last_test_day = all_days[ltd+seq_length]
    print(f'Standardization data are from {first_st_day} to {last_st_day}')
    print(f'Training data are from {first_train_day} to {last_train_day}')
    print(f'Testing data are from {first_test_day} to {last_test_day}')
    data_st = data[(data['datadate'] >= first_st_day) & (data['datadate'] <= last_st_day)].reset_index(drop=True)
    data_train = data[(data['datadate'] >= first_train_day) & (data['datadate'] <= last_train_day)].reset_index(drop=True)
    data_test = data[(data['datadate'] >= first_test_day) & (data['datadate'] <= last_test_day)].reset_index(drop=True)
    
    # Standardize here 
    # for both train and test, use the X days before the first train day
    df_mean = data_st[factors + ['tic']].groupby('tic').mean().reset_index()
    df_std = data_st[factors + ['tic']].groupby('tic').std().reset_index()
    df_train = pd.merge(data_train, df_mean, how='left', on=['tic'], suffixes=('', '_MEAN'))
    df_test = pd.merge(data_test, df_mean, how='left', on=['tic'], suffixes=('', '_MEAN'))
    for factor in factors:
        df_train[factor] = df_train[factor] - df_train[f'{factor}_MEAN']
        df_test[factor] = df_test[factor] - df_test[f'{factor}_MEAN']
    df_train = df_train[['datadate', 'tic'] + factors + ['ret_d', 'TBill1y', 'rel_ret_d', 'rank', 'sector']]
    df_test = df_test[['datadate', 'tic'] + factors + ['ret_d', 'TBill1y', 'rel_ret_d', 'rank', 'sector']]
    df_train = pd.merge(df_train, df_std, how='left', on=['tic'], suffixes=('', '_STD'))
    df_test = pd.merge(df_test, df_std, how='left', on=['tic'], suffixes=('', '_STD'))

    for factor in factors:
        df_train[factor] = df_train[factor] / df_train[f'{factor}_STD']
        df_test[factor] = df_test[factor] / df_test[f'{factor}_STD']
    data_train = df_train[['datadate', 'tic'] + factors + ['ret_d', 'TBill1y', 'rel_ret_d', 'rank', 'sector']]
    data_test = df_test[['datadate', 'tic'] + factors + ['ret_d', 'TBill1y', 'rel_ret_d', 'rank', 'sector']]

    # Fill NA
    for factor in factors:
        data_train.loc[:, factor] = data_train[factor].fillna(0)
        data_test.loc[:, factor] = data_test[factor].fillna(0)

    # Compute how many training data we will have
    all_train_days = list(data_train.datadate.unique())
    all_test_days = list(data_test.datadate.unique())
    num_train_days = len(all_train_days)
    num_test_days = len(all_test_days)
    num_train_data = (num_train_days - seq_length + 1) * nt
    num_test_data = (num_test_days - seq_length + 1) * nt
        
    # Create training data
    x_train = np.zeros((num_train_data, len(factors), seq_length))
    y_train = np.zeros((num_train_data, ))
    ret_d_train = np.zeros((num_train_data, ))
    sector_train = np.zeros((num_train_data, ))
    for i in range(num_train_days - seq_length + 1):
        train_days = all_train_days[i : seq_length + i]
        data_temp = data_train[data_train['datadate'].isin(train_days)]
        # Convert dataframe data to three dimensional training data (ticker, factor, time-series data)
        pivot_data = data_temp[factors+['datadate', 'tic']].pivot_table(index='tic', columns='datadate')
        x_train[i*nt:(i+1)*nt, :, :] = pivot_data.values.reshape(nt, len(factors), seq_length)
        y_train[i*nt:(i+1)*nt] = data_train[data_train['datadate'] == train_days[-1]]['rank'].values.reshape(nt, )
        ret_d_train[i*nt:(i+1)*nt] = data_train[data_train['datadate'] == train_days[-1]]['ret_d'].values.reshape(nt, )
        # Get categorical input sector
        sector_train[i*nt:(i+1)*nt] = data_train[data_train['datadate'] == train_days[-1]]['sector'].values.reshape(nt, )

    # Create testing data
    x_test = np.zeros((num_test_data, len(factors), seq_length))
    y_test = np.zeros((num_test_data, ))
    ret_d_test = np.zeros((num_test_data, ))
    sector_test = np.zeros((num_test_data, ))
    for i in range(num_test_days - seq_length + 1):
        test_days = all_test_days[i : seq_length + i]
        data_temp = data_test[data_test['datadate'].isin(test_days)]
        pivot_data = data_temp[factors+['datadate', 'tic']].pivot_table(index='tic', columns='datadate')
        x_test[i*nt:(i+1)*nt, :, :] = pivot_data.values.reshape(nt, len(factors), seq_length)
        y_test[i*nt:(i+1)*nt] = data_test[data_test['datadate'] == test_days[-1]]['rank'].values.reshape(nt, )
        ret_d_test[i*nt:(i+1)*nt] = data_test[data_test['datadate'] == test_days[-1]]['ret_d'].values.reshape(nt, )
        # Get categorical input sector
        sector_test[i*nt:(i+1)*nt] = data_test[data_test['datadate'] == test_days[-1]]['sector'].values.reshape(nt, )
    
    # Reshape train/test data so that it is channels_last
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))
    
    # Let the label start with 0 to align with sparse cross-entropy
    y_train[:] = y_train[:] + 2
    y_test[:] = y_test[:] + 2
    
    print(f'Training data have shape {x_train.shape}, {y_train.shape}')
    print(f'Testing data have shape {x_test.shape}, {y_test.shape}')
    
    return data_train, x_train, y_train, data_test, x_test, y_test, ret_d_train, ret_d_test, sector_train, sector_test

def prep_train_test_data_regression(data, seq_length, ftd, ltd, all_days, factors):
    '''
    Prepares the training/validation and test data for linear regression models
    '''

    num_st_days = 200 # Number of days used for standardization
    if ftd < num_st_days:
        print('not enough standardization days')
    
    first_st_day = all_days[ftd-num_st_days]
    last_st_day = all_days[ftd-1]
    first_train_day = all_days[ftd]
    last_train_day = all_days[ltd]
    first_test_day = all_days[ltd-seq_length+2]
    last_test_day = all_days[ltd+seq_length]
    print(f'Standardization data are from {first_st_day} to {last_st_day}')
    print(f'Training data are from {first_train_day} to {last_train_day}')
    print(f'Testing data are from {first_test_day} to {last_test_day}')
    data_st = data[(data['datadate'] >= first_st_day) & (data['datadate'] <= last_st_day)].reset_index(drop=True)
    data_train = data[(data['datadate'] >= first_train_day) & (data['datadate'] <= last_train_day)].reset_index(drop=True)
    data_test = data[(data['datadate'] >= first_test_day) & (data['datadate'] <= last_test_day)].reset_index(drop=True)
    
    # Standardize here 
    df_mean = data_st[factors + ['tic']].groupby('tic').mean().reset_index()
    df_std = data_st[factors + ['tic']].groupby('tic').std().reset_index()
    df_train = pd.merge(data_train, df_mean, how='left', on=['tic'], suffixes=('', '_MEAN'))
    df_test = pd.merge(data_test, df_mean, how='left', on=['tic'], suffixes=('', '_MEAN'))
    for factor in factors:
        df_train[factor] = df_train[factor] - df_train[f'{factor}_MEAN']
        df_test[factor] = df_test[factor] - df_test[f'{factor}_MEAN']
    df_train = df_train[['datadate', 'tic'] + factors + ['ret_d', 'TBill1y', 'rel_ret_d', 'rank', 'sector']]
    df_test = df_test[['datadate', 'tic'] + factors + ['ret_d', 'TBill1y', 'rel_ret_d', 'rank', 'sector']]
    df_train = pd.merge(df_train, df_std, how='left', on=['tic'], suffixes=('', '_STD'))
    df_test = pd.merge(df_test, df_std, how='left', on=['tic'], suffixes=('', '_STD'))
    for factor in factors:
        df_train[factor] = df_train[factor] / df_train[f'{factor}_STD']
        df_test[factor] = df_test[factor] / df_test[f'{factor}_STD']
    data_train = df_train[['datadate', 'tic'] + factors + ['ret_d', 'TBill1y', 'rel_ret_d', 'rank', 'sector']]
    data_test = df_test[['datadate', 'tic'] + factors + ['ret_d', 'TBill1y', 'rel_ret_d', 'rank', 'sector']]

    # Fill NA
    for factor in factors:
        data_train.loc[:, factor] = data_train[factor].fillna(0)
        data_test.loc[:, factor] = data_test[factor].fillna(0)
    
    y_train = np.array(data_train['ret_d'])
    y_test = np.array(data_test['ret_d'])

    return data_train, y_train, data_test, y_test

## Simulation Functions
def invest(day, day1, day20, day21, day22, num_stocks, total, total_asset, position_dict, top_stocks):
    '''
    Input:
        day: from 0 to (last_test_day - first_test_day - 1)
        day1: first_test_day
        day20: last_test_day
        day21: day of execution
        day22: day after execution to calculate what the holding amount will be
        num_stocks: the number of stocks to be traded daily
        total: total amount of holdings
        total_asset: sequence of total amount of holdings, starting with 1, next day's asset amount, ... etc
        position_dict: `num_stocks` items
            keys: tickers of current holdings
            values: amount of current holdings
        top_stocks: list of top `num_stocks` recommended stocks
    Output:
        total_asset: updated with one more entry
        total: updated with the new total, i.e., the last entry of the updated total_asset
        position_dict: positions on day22
    Calculates the total value of the portfolio when buying "top_stocks" on "day21" and hold until at least "day22"
    '''
    if day != 0:
        uninvested = 0
    elif (day == 0) & (first_run == False):
        uninvested = 0
    elif (day == 0) & (first_run == True):
        uninvested = 1
    to_hold = []
    to_delete = []

    for key in position_dict:
        # Sell
        if key not in top_stocks:
            uninvested += position_dict[key]
            to_delete.append(key)
            # print(f'Sell {key} on {day21} of market open')
        # Hold
        else:
            to_hold.append(key)

    # Remove sold stocks from position
    for key in to_delete:
        del position_dict[key]

    # print(f'Hold {to_hold} on {day21} market open')

    # Buy stocks
    for index in list(set(top_stocks) - set(to_hold)):
        position_dict[index] = uninvested / (num_stocks - len(to_hold))
        # print(f'Buy {index} on {day21} market open')

    # Calculate return right away
    for key in top_stocks:
        percent_change = data_test[(data_test['datadate'] == day20) & (data_test['tic'] == key)]['ret_d'].values[0]
        total += position_dict[key] * percent_change
        # This calculates the position on the next day
        position_dict[key] = position_dict[key] * (1 + percent_change)
    # print(f'position on {day22} market open will be {position_dict}')

    total_asset.append(total)
    print(f'Total asset on {day22} will be {total}')
    # print(f'It should be the same as {sum(position_dict.values())}')
    
    return total_asset, total, position_dict

def simulate(ftd, ltd, total_dict, first_run, num_stocks, total_asset_dict, position_dict_all, return_dict):
    '''
    Input:
        ftd: first train date
        ltd: last train date
        first_run: True or False that determines the initialization of 'uninvested'
        total_asset_dict:
            keys: from 0 to num_of_models + 1 (the last one being ensemble)
            values: sequence of total_asset starting with 1, next day's asset, ... etc; for plotting
        position_dict_all:
            keys: from 0 to num_of_models + 1
            values: dictionary of 5 items whose keys are the tickers of current holdings and values are the amount held
        num_stocks: number of stocks to pick
        return_dict: tracks the running 6-month return to be used for return-weighted ensemble
    Output:
        total_asset_dict: updated in the function
        total_dict:
            keys: from 0 to num_of_models + 1
            values: current asset amount; for double checking if asset calculation is correct
    Simulates the daily trading strategy over a testing period containing "seq_length" days
    '''
    if not first_run:
        y_score_weights = [0]*num_of_models
        for i in range(num_of_models):
            y_score_weights[i] = math.exp(return_dict[i][-1])
        sum_score_weights = sum(y_score_weights)
        for i in range(num_of_models):
            y_score_weights[i] /= sum_score_weights
            # print(f'Weight of model {i} is {y_score_weights[i]}')

    for day in range(seq_length):
        
        day1 = all_days[ltd - seq_length + 2 + day]
        day20 = all_days[ltd - seq_length + 2 + day + seq_length - 1]
        day21 = all_days[ltd - seq_length + 2 + day + seq_length]
        day22 = all_days[ltd - seq_length + 2 + day + seq_length + 1]

        data_test_temp = data_test[(data_test['datadate'] >= day1) & (data_test['datadate'] <= day20)]
        x_test = np.zeros((nt, len(factors), seq_length))
        y_test = np.zeros((nt, ))
        ret_d_test = np.zeros((nt, ))
        sector_test = np.zeros((nt, ))

        pivot_data = data_test_temp[factors+['datadate', 'tic']].pivot_table(index='tic', columns='datadate')
        x_test = pivot_data.values.reshape(nt, len(factors), seq_length)
        y_test[:] = data_test_temp[data_test_temp['datadate'] == day20]['rank'].values.reshape(nt, )
        ret_d_test = data_test_temp[data_test_temp['datadate'] == day20]['ret_d'].values.reshape(nt, )
        sector_test = data_test_temp[data_test_temp['datadate'] == day20]['sector'].values.reshape(nt, )

        x_test = np.transpose(x_test, (0, 2, 1))
        y_test[:] = y_test[:] + 2
        print(f'Testing data from {day1} to {day20} have shape {x_test.shape}, {y_test.shape}')

        y_scores = [0]*num_of_models
        for i in range(num_of_models):
            try:
                y_pred = CNN_model.model_dict[i].predict([y_test, x_test, ret_d_test, sector_test], batch_size=4096, verbose=0)
            except:
                y_pred = model_dict[i].predict([y_test, x_test, ret_d_test, sector_test], batch_size=4096, verbose=0)
            y_scores[i] = np.dot(y_pred, np.array([-2, -1, 0, 1, 2]))
            top_indices = np.argsort(y_scores[i])[-num_stocks:]
            top_stocks = [num_to_tic_dict[num] for num in top_indices]
            # print(f'top_stocks by model {i} to buy on {day21} are {top_stocks}')
            total_asset, total, position_dict = invest(day, day1, day20, day21, day22, num_stocks, total_dict[i], total_asset_dict[i], position_dict_all[i], top_stocks)
            total_dict[i] = total
            total_asset_dict[i] = total_asset
            position_dict_all[i] = position_dict
        
        if first_run:
            y_score_ensem = sum(y_scores) / num_of_models
        else:
            y_score_ensem = np.zeros((y_scores[0].shape))
            for i in range(num_of_models):
                y_score_ensem += y_scores[i] * y_score_weights[i]
                 
        top_indices = np.argsort(y_score_ensem)[-num_stocks:]
        top_stocks = [num_to_tic_dict[num] for num in top_indices]
        # print(f'top_stocks by return weighted ensemble to buy on {day21} are {top_stocks}')
        total_asset, total, position_dict = invest(day, day1, day20, day21, day22, num_stocks, total_dict['ensemble_weighted'], total_asset_dict['ensemble_weighted'], 
                                                    position_dict_all['ensemble_weighted'], top_stocks)
        total_dict['ensemble_weighted'] = total
        total_asset_dict['ensemble_weighted'] = total_asset
        position_dict_all['ensemble_weighted'] = position_dict

        y_score_ensem = sum(y_scores) / num_of_models
        top_indices = np.argsort(y_score_ensem)[-num_stocks:]
        top_stocks = [num_to_tic_dict[num] for num in top_indices]
        # print(f'top_stocks by equal weighted ensemble to buy on {day21} are {top_stocks}')
        total_asset, total, position_dict = invest(day, day1, day20, day21, day22, num_stocks, total_dict['ensemble_equal'], total_asset_dict['ensemble_equal'], 
                                                    position_dict_all['ensemble_equal'], top_stocks)
        total_dict['ensemble_equal'] = total
        total_asset_dict['ensemble_equal'] = total_asset
        position_dict_all['ensemble_equal'] = position_dict        

        
    # Calculate the return over the entire test period
    if first_run:
        return_dict = {}
        for i in range(num_of_models):
            return_dict[i] = []
    for i in range(num_of_models):
        # first and last day returns
        ldr = total_asset_dict[i][-1]
        fdr = total_asset_dict[i][max(-len(total_asset_dict[i]), -20*6-1)]
        ret = (ldr - fdr) / fdr
        return_dict[i].append(ret)
        # print(f'Model {i} changed {ret*100} percent from {all_days[ltd+23+max(-len(total_asset_dict[i]), -20*6-1)]} to {all_days[ltd+22]}')
    
    # +22 because +20 from testing; +1 from excluding right endpoint; +1 from one day look ahead;
    # +1 from switching from last train day to first test day
    # -1 from needing the first day where the starting asset 1
    if num_iter % 10 == 0 or num_iter == num_iters - 1:
        plt.figure(figsize=(16, 6))
        x_axis = all_days[ltd+22-len(total_asset_dict[0]):ltd+22]
        for i in range(num_of_models):
            plt.plot(x_axis, total_asset_dict[i], label=f'model {i}')
        plt.plot(x_axis, total_asset_dict['ensemble_weighted'], label='exp weighted ensemble')
        plt.plot(x_axis, total_asset_dict['ensemble_equal'], label='equal weighted ensemble')
        indices_to_display = np.linspace(0, len(x_axis)-1, 15, dtype=int)
        plt.xticks(indices_to_display, [x_axis[i] for i in indices_to_display], rotation=45)
        plt.grid()
        plt.legend()
        plt.show()

    return total_asset_dict, total_dict, position_dict_all, return_dict

def invest_decile(day, day1, day20, day21, day22, total, total_asset, position_dict, decile_stocks):
    '''
    The same as the "invest" function except that it is for decile investment and the calculation is more efficient
    '''
    if day != 0:
        uninvested = 0
    elif (day == 0) & (first_run == False):
        uninvested = 0
    elif (day == 0) & (first_run == True):
        uninvested = 1

    to_delete = set(position_dict.keys()) - decile_stocks
    to_hold = set(position_dict.keys()) - to_delete

    # Remove sold stocks from position
    for key in to_delete:
        uninvested += position_dict[key]
        del position_dict[key]

    # Buy stocks
    new_investment_amount = uninvested / (len(decile_stocks) - len(to_hold))
    for index in decile_stocks - to_hold:
        position_dict[index] = new_investment_amount

    # Calculate return right away
    all_returns = data_test[data_test['datadate'] == day20]['ret_d'].values
    for key in decile_stocks:
        percent_change = all_returns[tic_to_num_dict[key]]
        total += position_dict[key] * percent_change
        # This calculates the position on the next day
        position_dict[key] = position_dict[key] * (1 + percent_change)
    total_asset.append(total)
    
    return total_asset, position_dict

def simulate_decile(ftd, ltd, first_run, total_asset_dict, position_dict_all):
    '''
    Same as the "simulate" function except that this is for decile investment
    '''

    for day in range(seq_length):
        
        day1 = all_days[ltd - seq_length + 2 + day]
        day20 = all_days[ltd - seq_length + 2 + day + seq_length - 1]
        day21 = all_days[ltd - seq_length + 2 + day + seq_length]
        day22 = all_days[ltd - seq_length + 2 + day + seq_length + 1]

        data_test_temp = data_test[(data_test['datadate'] >= day1) & (data_test['datadate'] <= day20)]
        x_test = np.zeros((nt, len(factors), seq_length))
        y_test = np.zeros((nt, ))
        ret_d_test = np.zeros((nt, ))
        sector_test = np.zeros((nt, ))

        pivot_data = data_test_temp[factors+['datadate', 'tic']].pivot_table(index='tic', columns='datadate')
        x_test = pivot_data.values.reshape(nt, len(factors), seq_length)
        y_test[:] = data_test_temp[data_test_temp['datadate'] == day20]['rank'].values.reshape(nt, )
        ret_d_test = data_test_temp[data_test_temp['datadate'] == day20]['ret_d'].values.reshape(nt, )
        sector_test = data_test_temp[data_test_temp['datadate'] == day20]['sector'].values.reshape(nt, )

        x_test = np.transpose(x_test, (0, 2, 1))
        y_test[:] = y_test[:] + 2
        print(f'Testing data from {day1} to {day20} have shape {x_test.shape}, {y_test.shape}')

        y_scores = [0]*num_of_models
        for i in range(num_of_models):
            y_pred = model_dict[i].predict([y_test, x_test, ret_d_test, sector_test], batch_size=4096, verbose=0)
            y_scores[i] = np.dot(y_pred, np.array([-2, -1, 0, 1, 2]))

            # Calculate the return of individual models for each decile
            scores_order = np.argsort(y_scores[i])
            indices_size = int(len(scores_order) / 10)
            for j in range(10):
                decile_indices = scores_order[indices_size*j : indices_size*(j+1)]
                decile_stocks = set([num_to_tic_dict[num] for num in decile_indices])
                total_asset, position_dict = invest_decile(day, day1, day20, day21, day22, total_asset_dict[f'model_{i}_decile_{j}'][-1], 
                                                           total_asset_dict[f'model_{i}_decile_{j}'], position_dict_all[f'model_{i}_decile_{j}'], decile_stocks)
                total_asset_dict[f'model_{i}_decile_{j}'] = total_asset
                position_dict_all[f'model_{i}_decile_{j}'] = position_dict

        # Calculate the return of equal-weighted ensembles for each decile
        y_score_ensem = sum(y_scores) / num_of_models
        scores_order = np.argsort(y_score_ensem)
        for j in range(10):
            decile_indices = scores_order[indices_size*j : indices_size*(j+1)]
            decile_stocks = set([num_to_tic_dict[num] for num in decile_indices])
            total_asset, position_dict = invest_decile(day, day1, day20, day21, day22, total_asset_dict[f'ensemble_equal_decile_{j}'][-1], 
                                                       total_asset_dict[f'ensemble_equal_decile_{j}'], position_dict_all[f'ensemble_equal_decile_{j}'], decile_stocks)
            total_asset_dict[f'ensemble_equal_decile_{j}'] = total_asset
            position_dict_all[f'ensemble_equal_decile_{j}'] = position_dict
    
    # +23 because +20 from testing; +1 from excluding right endpoint; +1 from one day look ahead;
    # +1 from switching from last train day to first test day
    # -1 from needing the first day where the starting asset 1
    if num_iter % 20 == 0 or num_iter == num_iters - 1:
        x_axis = all_days[ltd+22-len(total_asset_dict['model_0_decile_0']):ltd+22]
        for j in range(10):
            plt.figure(figsize=(16, 6))
            for i in range(num_of_models):
                plt.plot(x_axis, total_asset_dict[f'model_{i}_decile_{j}'], label=f'model {i}')
            plt.plot(x_axis, total_asset_dict[f'ensemble_equal_decile_{j}'], label=f'Ensemble')
            indices_to_display = np.linspace(0, len(x_axis)-1, 15, dtype=int)
            plt.xticks(indices_to_display, [x_axis[i] for i in indices_to_display], rotation=45)
            plt.grid()
            plt.legend()
            plt.show()

    return total_asset_dict, position_dict_all

def invest_top_bot(day, day1, day20, day21, day22, num_stocks, total, total_asset, position_dict, top_bot_stocks):
    '''
    Same as "invest" except this calculates the portfolio value for both top and bottom 10 stocks; also serves as a sanity check
    '''
    if day != 0:
        uninvested = 0
    elif (day == 0) & (first_run == False):
        uninvested = 0
    elif (day == 0) & (first_run == True):
        uninvested = 1
    to_hold = []
    to_delete = []

    for key in position_dict:
        # Sell
        if key not in top_bot_stocks:
            uninvested += position_dict[key]
            to_delete.append(key)
            # print(f'Sell {key} on {day21} of market open')
        # Hold
        else:
            to_hold.append(key)

    # Remove sold stocks from position
    for key in to_delete:
        del position_dict[key]

    # print(f'Hold {to_hold} on {day21} market open')

    # Buy stocks
    for index in list(set(top_bot_stocks) - set(to_hold)):
        position_dict[index] = uninvested / (num_stocks - len(to_hold))
        # print(f'Buy {index} on {day21} market open')

    # Calculate return right away
    for key in top_bot_stocks:
        percent_change = data_test[(data_test['datadate'] == day20) & (data_test['tic'] == key)]['ret_d'].values[0]
        total += position_dict[key] * percent_change
        # This calculates the position on the next day
        position_dict[key] = position_dict[key] * (1 + percent_change)
    # print(f'position on {day22} market open will be {position_dict}')

    total_asset.append(total)
    print(f'Total asset on {day22} will be {total}')
    # print(f'It should be the same as {sum(position_dict.values())}')
    
    return total_asset, total, position_dict

def simulate_top_bot(ftd, ltd, total_dict, first_run, num_stocks, total_asset_dict, position_dict_all, return_dict):
    '''
    Same as "simulate" except this calculates the portfolio value for both top and bottom 10 stocks; also serves as a sanity check
    '''
    if not first_run:
        y_score_weights_top = [0]*num_of_models
        y_score_weights_bot = [0]*num_of_models
        for i in range(num_of_models):
            y_score_weights_top[i] = math.exp(return_dict[f'top_{i}'][-1])
            y_score_weights_bot[i] = math.exp(-return_dict[f'bot_{i}'][-1])
        sum_score_weights_top = sum(y_score_weights_top)
        sum_score_weights_bot = sum(y_score_weights_bot)
        for i in range(num_of_models):
            y_score_weights_top[i] /= sum_score_weights_top
            y_score_weights_bot[i] /= sum_score_weights_bot

    for day in range(seq_length):
        
        day1 = all_days[ltd - seq_length + 2 + day]
        day20 = all_days[ltd - seq_length + 2 + day + seq_length - 1]
        day21 = all_days[ltd - seq_length + 2 + day + seq_length]
        day22 = all_days[ltd - seq_length + 2 + day + seq_length + 1]

        data_test_temp = data_test[(data_test['datadate'] >= day1) & (data_test['datadate'] <= day20)]
        x_test = np.zeros((nt, len(factors), seq_length))
        y_test = np.zeros((nt, ))
        ret_d_test = np.zeros((nt, ))
        sector_test = np.zeros((nt, ))

        pivot_data = data_test_temp[factors+['datadate', 'tic']].pivot_table(index='tic', columns='datadate')
        x_test = pivot_data.values.reshape(nt, len(factors), seq_length)
        y_test[:] = data_test_temp[data_test_temp['datadate'] == day20]['rank'].values.reshape(nt, )
        ret_d_test = data_test_temp[data_test_temp['datadate'] == day20]['ret_d'].values.reshape(nt, )
        sector_test = data_test_temp[data_test_temp['datadate'] == day20]['sector'].values.reshape(nt, )

        x_test = np.transpose(x_test, (0, 2, 1))
        y_test[:] = y_test[:] + 2
        print(f'Testing data from {day1} to {day20} have shape {x_test.shape}, {y_test.shape}')

        y_scores = [0]*num_of_models
        for i in range(num_of_models):
            y_pred = model_dict[i].predict([y_test, x_test, ret_d_test, sector_test], batch_size=4096, verbose=0)
            y_scores[i] = np.dot(y_pred, np.array([-2, -1, 0, 1, 2]))

            # top 10 stocks
            top_indices = np.argsort(y_scores[i])[-num_stocks:]
            top_stocks = [num_to_tic_dict[num] for num in top_indices]
            # print(f'top_stocks by model {i} to buy on {day21} are {top_stocks}')
            total_asset, total, position_dict = invest_top_bot(day, day1, day20, day21, day22, num_stocks, total_dict[f'model_{i}_top'], total_asset_dict[f'model_{i}_top'], 
                                                        position_dict_all[f'model_{i}_top'], top_stocks)
            total_dict[f'model_{i}_top'] = total
            total_asset_dict[f'model_{i}_top'] = total_asset
            position_dict_all[f'model_{i}_top'] = position_dict

            # bot 10 stocks
            bot_indices = np.argsort(y_scores[i])[:num_stocks]
            bot_stocks = [num_to_tic_dict[num] for num in bot_indices]
            total_asset, total, position_dict = invest_top_bot(day, day1, day20, day21, day22, num_stocks, total_dict[f'model_{i}_bot'], total_asset_dict[f'model_{i}_bot'],
                                                        position_dict_all[f'model_{i}_bot'], bot_stocks)
            total_dict[f'model_{i}_bot'] = total
            total_asset_dict[f'model_{i}_bot'] = total_asset
            position_dict_all[f'model_{i}_bot'] = position_dict
        
        if first_run:
            y_score_ensem_top = sum(y_scores) / num_of_models
            y_score_ensem_bot = sum(y_scores) / num_of_models
        else:
            y_score_ensem_top = np.zeros((y_scores[0].shape))
            y_score_ensem_bot = np.zeros((y_scores[0].shape))
            for i in range(num_of_models):
                y_score_ensem_top += y_scores[i] * y_score_weights_top[i]
                y_score_ensem_bot += y_scores[i] * y_score_weights_bot[i]
        
        # top 10 stocks
        top_indices = np.argsort(y_score_ensem_top)[-num_stocks:]
        top_stocks = [num_to_tic_dict[num] for num in top_indices]
        total_asset, total, position_dict = invest_top_bot(day, day1, day20, day21, day22, num_stocks, total_dict['ensemble_weighted_top'], total_asset_dict['ensemble_weighted_top'], 
                                                    position_dict_all['ensemble_weighted_top'], top_stocks)
        total_dict['ensemble_weighted_top'] = total
        total_asset_dict['ensemble_weighted_top'] = total_asset
        position_dict_all['ensemble_weighted_top'] = position_dict

        y_score_ensem = sum(y_scores) / num_of_models
        top_indices = np.argsort(y_score_ensem)[-num_stocks:]
        top_stocks = [num_to_tic_dict[num] for num in top_indices]
        total_asset, total, position_dict = invest_top_bot(day, day1, day20, day21, day22, num_stocks, total_dict['ensemble_equal_top'], total_asset_dict['ensemble_equal_top'], 
                                                    position_dict_all['ensemble_equal_top'], top_stocks)
        total_dict['ensemble_equal_top'] = total
        total_asset_dict['ensemble_equal_top'] = total_asset
        position_dict_all['ensemble_equal_top'] = position_dict

        # bot 10 stocks
        bot_indices = np.argsort(y_score_ensem_bot)[:num_stocks]
        bot_stocks = [num_to_tic_dict[num] for num in bot_indices]
        total_asset, total, position_dict = invest_top_bot(day, day1, day20, day21, day22, num_stocks, total_dict['ensemble_weighted_bot'], total_asset_dict['ensemble_weighted_bot'], 
                                                    position_dict_all['ensemble_weighted_bot'], bot_stocks)
        total_dict['ensemble_weighted_bot'] = total
        total_asset_dict['ensemble_weighted_bot'] = total_asset
        position_dict_all['ensemble_weighted_bot'] = position_dict

        y_score_ensem = sum(y_scores) / num_of_models
        bot_indices = np.argsort(y_score_ensem)[:num_stocks]
        bot_stocks = [num_to_tic_dict[num] for num in bot_indices]
        total_asset, total, position_dict = invest_top_bot(day, day1, day20, day21, day22, num_stocks, total_dict['ensemble_equal_bot'], total_asset_dict['ensemble_equal_bot'], 
                                                    position_dict_all['ensemble_equal_bot'], bot_stocks)
        total_dict['ensemble_equal_bot'] = total
        total_asset_dict['ensemble_equal_bot'] = total_asset
        position_dict_all['ensemble_equal_bot'] = position_dict
        
    # Calculate the return over the entire test period
    if first_run:
        return_dict = {}
        for i in range(num_of_models):
            return_dict[f'top_{i}'] = []
            return_dict[f'bot_{i}'] = []
    for i in range(num_of_models):
        # first and last day returns
        ldr = total_asset_dict[f'model_{i}_top'][-1]
        fdr = total_asset_dict[f'model_{i}_top'][max(-len(total_asset_dict[f'model_{i}_top']), -20*6-1)]
        ret = (ldr - fdr) / fdr
        return_dict[f'top_{i}'].append(ret)
        ldr = total_asset_dict[f'model_{i}_bot'][-1]
        fdr = total_asset_dict[f'model_{i}_bot'][max(-len(total_asset_dict[f'model_{i}_bot']), -20*6-1)]
        ret = (ldr - fdr) / fdr
        return_dict[f'bot_{i}'].append(ret)
        # print(f'Model {i} changed {ret*100} percent from {all_days[ltd+23+max(-len(total_asset_dict[i]), -20*6-1)]} to {all_days[ltd+22]}')
    
    if num_iter % 20 == 0 or num_iter == num_iters - 1: 
        plt.figure(figsize=(16, 6))
        x_axis = all_days[ltd+22-len(total_asset_dict['model_0_top']):ltd+22]
        for i in range(num_of_models):
            plt.plot(x_axis, total_asset_dict[f'model_{i}_top'], label=f'model {i} top')
        plt.plot(x_axis, total_asset_dict['ensemble_weighted_top'], label='exp weighted ensemble top')
        plt.plot(x_axis, total_asset_dict['ensemble_equal_top'], label='equal weighted ensemble top')
        indices_to_display = np.linspace(0, len(x_axis)-1, 15, dtype=int)
        plt.xticks(indices_to_display, [x_axis[i] for i in indices_to_display], rotation=45)
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure(figsize=(16, 6))
        x_axis = all_days[ltd+22-len(total_asset_dict['model_0_bot']):ltd+22]
        for i in range(num_of_models):
            plt.plot(x_axis, total_asset_dict[f'model_{i}_bot'], label=f'model {i} bot')
        plt.plot(x_axis, total_asset_dict['ensemble_weighted_bot'], label='exp weighted ensemble bot')
        plt.plot(x_axis, total_asset_dict['ensemble_equal_bot'], label='equal weighted ensemble bot')
        indices_to_display = np.linspace(0, len(x_axis)-1, 15, dtype=int)
        plt.xticks(indices_to_display, [x_axis[i] for i in indices_to_display], rotation=45)
        plt.grid()
        plt.legend()
        plt.show()

    return total_asset_dict, total_dict, position_dict_all, return_dict

## CNN Class
scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

class CNN:
    
    def __init__(self, input_shape, seed, **kwargs):
        
        self.__dict__.update(kwargs)
        
        self.input_shape = input_shape
        self.seed = seed
        self.train_callback = [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=self.plateau_patience, min_lr=self.min_learning_rate),
                               keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.train_patience, min_delta=self.min_delta,
                                                             restore_best_weights=True, verbose=1)]
        self.retrain_callback = [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=self.plateau_patience, min_lr=self.min_learning_rate),
                                 keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.retrain_patience, min_delta=self.min_delta,
                                                               restore_best_weights=True, verbose=1)]
        
        self.model_dict = self.build_model()
    
    def build_base_model(self, output_dim, seed):
        init = tf.keras.initializers.HeNormal(seed)
        
        input_layer = keras.layers.Input(self.input_shape)
        x = input_layer

        # Use sector embedding as an embedding to add to the input data
        embedding_layer = Embedding(input_dim=self.num_of_tokens, output_dim=self.embedding_dim, 
                                    )(self.sector_input)
        embedding_layer = tf.tile(embedding_layer, [1, 20, 1])
        x = Add()([x, embedding_layer])
        
        assert len(self.filter_dims) == len(self.kernel_sizes)
        assert len(self.filter_dims) == len(self.strides)
        assert len(self.filter_dims) == len(self.paddings)

        for i in range(len(self.filter_dims)):
            x = Conv1D(filters=self.filter_dims[i], kernel_size=self.kernel_sizes[i], padding=self.paddings[i], strides=self.strides[i],
                       kernel_initializer=init, 
                       use_bias = False
                       )(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout(self.dropout_conv, seed=self.seed)(x)
        
        x = GlobalAveragePooling1D()(x)

        for layer_dim in self.layer_dims:
            x = Dense(layer_dim, activation=self.activation, kernel_initializer=init, bias_initializer='zeros')(x)
            x = Dropout(self.dropout_dense, seed=self.seed)(x)
        output_layer = Dense(output_dim, activation="softmax", kernel_initializer=init, use_bias=False)(x)
        model = keras.models.Model(inputs=[self.target, input_layer, self.ret_d, self.sector_input], outputs=output_layer)
        model.add_loss(self.custom_loss(self.target, output_layer, self.ret_d))

        return model

    def build_model(self):
        
        self.model_dict = {}

        for i in range(self.num_models):
            
            self.model_dict[i] = self.build_base_model(5, self.seed+i)

        return self.model_dict
    
    def custom_loss(self, y_true, y_pred, ret_d):
        y_true = tf.cast(y_true, dtype=tf.float32)
        return tf.reduce_mean(scce(y_true, y_pred) * tf.minimum(tf.abs(ret_d), 0.5))
    
    # Numpy version of loss
    def custom_loss_np(self, y_true, y_pred, ret_d):
        return np.mean(scce(y_true, y_pred) * np.minimum(abs(ret_d), 0.5))

    def compile_model(self):

        for i in range(self.num_models):
            self.model_dict[i].compile(loss=None, optimizer=keras.optimizers.Adam(self.learning_rate))

        # Just see one model architecture because all of them are the same
        self.model_dict[0].summary()
    
    # Using the same code for training and retraining model
    def train_model(self, x_train, y_train, ret_d_train, sector_train):
        tf.random.set_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        for i in range(self.num_models):
            history = self.model_dict[i].fit(
                x=[y_train, x_train, ret_d_train, sector_train],
                y=None,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=self.train_callback,
                validation_split=self.validation_split,
                verbose=self.verbose
                )
            gc.collect()
    
    def evaluate_model(self, x_train, y_train, ret_d_train, sector_train, x_test, y_test, ret_d_test, sector_test, batch_size):
        for i in range(self.num_models):
            y_pred = self.model_dict[i].predict([y_train, x_train, ret_d_train, sector_train], batch_size=batch_size)
            print(f'Model {i} training loss {self.custom_loss_np(y_train, y_pred, ret_d_train)}')
            y_pred = self.model_dict[i].predict([y_test, x_test, ret_d_test, sector_test], batch_size=batch_size)
            print(f'Model {i} test loss {self.custom_loss_np(y_test, y_pred, ret_d_test)}')
        
    def retrain_model(self, x_train, y_train, ret_d_train, sector_train):
        tf.random.set_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        for i in range(self.num_models):
            # Set learning rate back
            K.set_value(self.model_dict[i].optimizer.learning_rate, self.retrain_learning_rate)

            # Retrain models
            history = self.model_dict[i].fit(
                x=[y_train, x_train, ret_d_train, sector_train],
                y=None,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=self.retrain_callback,
                validation_split=self.validation_split,
                verbose=self.verbose
                )
            gc.collect()