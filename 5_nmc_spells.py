# -*- coding: utf-8 -*-
"""
Author: Joe King
Date: 15 Dec. 2019

    Script identifies helicopter maintenance spells by grouping by serial number and elapsed time between
    faults. If two fault >60min these are considered separate maintenance spells, otherwise considered
    part of same maintenance work.

"""

import pandas as pd
import numpy as np
import os, warnings
pd.options.display.max_rows = 150
pd.options.display.max_columns = 75
pd.options.mode.chained_assignment = None # turns of SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def wide_long(df, id_vars, to_unpivot, freq='H'):
    '''
    Function converts a pandas df from wide to long form. Returns long-form dataframe.
    :id_vars: list of ID variables to uniquely identify a row or record
    :to_unpivot: list of two date variables in wide format to convert to long
    :freq: str for frequency to forward fill, see pd.ffill() for details

    '''
    print("\nConverting wide to long..")
    d = pd.melt(df, id_vars=id_vars, value_vars=to_unpivot, value_name='date').sort_values(by=id_vars).reset_index(drop=True)
    d = d.set_index('date').groupby(id_vars).resample(freq).ffill()[[]].reset_index(drop=False).drop(columns=['id'])
    d.drop_duplicates(inplace=True)
    d['date'] = pd.to_datetime(d['date'], format ='%Y-%m-%d %H:%M', errors='raise')
    return d

def holiday(col):
    '''
    Checks whether date in a given column is a business work day, as determined by whether
        it's not in list of US holidays or weekends.
    :col: Pandas series
    '''
    cal = calendar()
    holidays = pd.Series(cal.holidays(start='2010-10-01', end='2019-12-31')).dt.date # list of US federal holidays in date range
    return col.isin(holidays)

def create_spells(df):
    '''
    Function creates maintenance spells using wide-form fault data. First converts to long-form dataset. Next,
        identifies spells as maintenance periods with >1 hour gap and calculates start and end time as min and max datetime.
        Aggregates covariate information by spells. Calculates nmc duration. Calculates proxy for repair duration
        as number of non-holiday+drill days between hours of 8-17h, summing these by spell.
    :param df: wide-form pandas dataframe of faults
    :return: df of serno-spell_id observations and long-form dataframe of SERNO x hour obs.
    '''
    reduced = df.loc[:, ['SERNO', 'fault_bgn_hr', 'fault_end_hr']].drop_duplicates().reset_index(drop=True)

    # wide to long conversion
    reduced['id'] = reduced.groupby(
        'SERNO').cumcount()  # uniquely identify each fault per serial number, ngroup is alternative
    long = wide_long(reduced, id_vars=['SERNO', 'id'], to_unpivot=['fault_bgn_hr', 'fault_end_hr'])
    long.sort_values(by=['SERNO', 'date'], inplace=True)
    long['delta'] = (long['date'] - long.groupby('SERNO')['date'].shift(1)) / np.timedelta64(1, 'h')  # hrs between this row and next row
    long['flag'] = np.where(long['delta'] > 1, 1, 0)
    long['spell_id'] = long.groupby('SERNO')['flag'].cumsum()
    long['spell_bgn'] = long.groupby(['SERNO', 'spell_id'])['date'].transform('min')
    long['spell_end'] = long.groupby(['SERNO', 'spell_id'])['date'].transform('max')
    long.drop(columns=['flag', 'delta'], inplace=True)

    print("\nCreating spell covariates..")

    # flag for whether day is holiday
    long['holiday'] = holiday(long['date'].dt.date)

    # day of week
    long['dayname'] = long['date'].dt.day_name()

    # faults opened per hour
    faults_started = df[['SERNO', 'fault_bgn_hr', 'PHASE', 'CWUC', 'dom_topic', 'phase1', 'phase2']]
    faults_started['ct'] = 1 # flag to allow count of faults in groupby
    faults_started = faults_started.groupby(['SERNO', 'fault_bgn_hr']).agg({'ct': ['count'], 'PHASE': ['max'],
                        'phase1': ['max'], 'phase2': ['max'], 'CWUC': ['unique'], 'dom_topic': ['unique']}).reset_index()
    faults_started.columns = ['SERNO', 'date', 'nr_faults', 'PHASE', 'phase1', 'phase2', 'CWUC', 'dom_topic'] # rename cols from MultiIndex

    # faults closed per hour
    faults_ended = df[['SERNO', 'fault_end_hr']]
    faults_ended['ct'] = 1  # flag to allow count of faults in groupby
    faults_ended = faults_ended.groupby(['SERNO', 'fault_end_hr']).agg({'ct': ['count']}).reset_index() # only want fault closed to identify drill weekend
    faults_ended.columns = ['SERNO', 'date', 'nr_faults_end']

    faults_agg = pd.concat([faults_started, faults_ended], ignore_index=True); del faults_started, faults_ended

    # merge fault info into long dataset
    long = long.merge(faults_agg, on=['SERNO', 'date'], how='left')
    long['nr_faults'].replace(np.NaN, 0, inplace=True)
    long['nr_faults_end'].replace(np.NaN, 0, inplace=True)

    # identify drill weekends
    long['wknd'] = np.where(long['dayname'].isin(['Saturday', 'Sunday']), 1, 0)
    long['wknd_ct'] = long['date'].dt.year.astype(str) + '-' + long['date'].dt.week.astype(str) # uniquely identify weekends in year
    long['drill'] = np.where((long['wknd']==1) &
            ((long['nr_faults']+long['nr_faults_end'])>0), 1, 0) # identify if fault opened/closed on weekend
    long['drill'] = long.groupby(['SERNO', 'wknd_ct', 'wknd'])['drill'].transform('max') # apply drill flag to whole weekend if fault opened/closed either day
    long.drop(columns=['nr_faults_end', 'wknd_ct'], inplace=True)

    # count valid hrs for all hours on business days + drill weekends
    # originally also used (long['date'].dt.hour>=7) & (long['date'].dt.hour<=20)
    long['valid'] = np.where(((long['wknd']==0) | (long['drill']==1)) & (long['holiday']==False), 1, 0)

    # correct valid hours flag to ensure max 8 valid flags (valid work hours) per spell per day
    long['temp'] = long.date.dt.date
    long['valid_ct'] = long.loc[long.valid==1].groupby(['SERNO', 'spell_id', 'temp'])['valid'].cumcount() + 1 # add one to fix zero-based indexing
    long['valid'] = np.where(long['valid_ct']<=8, long['valid'], 0) # no longer count as valid work hrs after 8 per day

    long.drop(columns=['wknd', 'drill', 'holiday', 'temp', 'valid_ct'], inplace=True)

    # aggregate repair info per spell
    for _ in ['PHASE', 'phase1', 'phase2']:
        long[_] = long.groupby(['SERNO', 'spell_id'])[_].transform('max')
    long['nr_faults'] = long.groupby(['SERNO', 'spell_id'])['nr_faults'].cumsum()
    long['nr_faults'] = long.groupby(['SERNO', 'spell_id'])['nr_faults'].transform('max')

    # convert pandas list to string, count number of unique codes and modal codes
    for col in ['CWUC', 'dom_topic']:
        agg = long[['SERNO', 'spell_id', col]]
        agg = agg.dropna(subset=[col])
        agg[col] = agg[col].apply(lambda x: ' '.join(map(str, x))) # unpack pd Series of lists to string
        agg = agg.groupby(['SERNO', 'spell_id'])[col].apply(' '.join).reset_index() # combine strings into one row per SERNO-spell_id
        agg[col+'_nuniq'] = agg[col].apply(lambda x: len(set(x.split()))) # number of unique codes per SERNO-spell_id
        agg[col+'_mode'] = agg[col].apply(lambda x: max(set(x.split()), key=x.count)) # get modal category
        long = long.merge(agg[['SERNO', 'spell_id', col+'_nuniq', col+'_mode']], on=['SERNO', 'spell_id'], how='left')
    long.drop(columns=['CWUC', 'dom_topic'], inplace=True)

    '''
    TODO: repair duration in hours. I believe we want to count the number of distinct valid hours within each spell
    correcting spells that start and end in same hour helps, but additional checks needed. 1-hour duration spells often 
    lapse into/out of holidays
    '''

    long['repair_duration'] = long.groupby(['SERNO', 'spell_id'])['valid'].transform('sum') # subtract 1 from this for those >=2 (verify)
    long.loc[long['spell_end'] == long['spell_bgn'], 'repair_duration'] = 0 # correction for spells < 60 min, else will be 0, 1, or 2 hrs

    spells = long.drop_duplicates(subset=['SERNO', 'spell_id'], keep='first').reset_index(drop=True).drop(columns=['spell_id', 'date', 'valid'])
    spells.rename(columns={'dayname': 'bgn_day'}, inplace= True)

    return long, spells

if __name__ == '__main__':

    #####################################################################
    #####    Read data, round fault times, convert long to wide     #####
    #####################################################################

    directory = r'C:\Users\jking\Documents\FTS2B\data'
    dir_shared = r'X:\Pechacek\ARNG Aircraft Readiness\Data\Processed\Master fault data intermediate\NGB Blackhawk'

    f = pd.read_csv(os.path.join(directory, 'NMC_faults_lda.csv'))

    # Union of PHASE flag
    f['PHASE'] = f['PHASE'] + f['phase_fault']
    del f['phase_fault']

    # set fault begin & end as datetime, round to nearest whole hour
    f['fault_bgn'] = pd.to_datetime(f['fault_bgn'], format ='%Y-%m-%d %H:%M', errors='raise')
    f['fault_end'] = pd.to_datetime(f['fault_end'], format ='%Y-%m-%d %H:%M', errors='raise')
    f['fault_bgn_hr'] = f['fault_bgn'].dt.round("H")  # round to nearest hour
    f['fault_end_hr'] = f['fault_end'].dt.round("H")

    summary_stats = pd.DataFrame(data={'Description': ['Total NMC faults'], 'N': [len(f)]})

    # identify faults with duration of 0 minutes and drop
    f['fault_duration'] = (f['fault_end'] - f['fault_bgn']).astype('timedelta64[m]')
    f = f.loc[f.fault_duration > 0].reset_index(drop=True)
    del f['fault_duration']

    # Forward fill missing flight hours
    f.sort_values(by=['SERNO', 'fault_bgn_hr', 'fault_end_hr'], inplace=True)
    f['FLT_HRS'] = f.groupby('SERNO')['FLT_HRS'].ffill()

    # get long-form version of faults and spells
    long, spells = create_spells(f)

    summary_stats = summary_stats.append({'Description': 'Total long, filled SERNO hours', 'N': len(long)},
                                         ignore_index=True)

    summary_stats = summary_stats.append({'Description': 'Total unique spells', 'N': len(spells)}, ignore_index=True)


    # merge modal flight hours by hour (from fault start date) onto spell dataset
    flt_hrs = f.groupby(['SERNO', 'fault_bgn_hr'])['FLT_HRS'].apply(lambda x:  pd.Series.mode(x)[0]).reset_index()
    spells = spells.merge(flt_hrs, left_on=['SERNO', 'spell_bgn'], right_on=['SERNO', 'fault_bgn_hr'], how='left')
    del spells['fault_bgn_hr']

    # create nmc duration in minutes measure
    bgn = f.groupby(['SERNO', 'fault_bgn_hr'])['fault_bgn'].min().reset_index()
    end = f.groupby(['SERNO', 'fault_end_hr'])['fault_end'].max().reset_index()

    spells = spells.merge(bgn, left_on=['SERNO', 'spell_bgn'], right_on=['SERNO', 'fault_bgn_hr'], how='left')
    spells = spells.merge(end, left_on=['SERNO', 'spell_end'], right_on=['SERNO', 'fault_end_hr'], how='left')
    spells['spell_bgn'] = np.where(spells['fault_bgn'].notnull(), spells['fault_bgn'], spells['spell_bgn']) # replace spell begin/end with fault datetimes
    spells['spell_end'] = np.where(spells['fault_end'].notnull(), spells['fault_end'], spells['spell_end']) # if no match, keep rounded version
    spells = spells.iloc[:, :-4]

    # nmc duration in minutes
    spells['nmc_duration'] = (spells['spell_end'] - spells['spell_bgn']).astype('timedelta64[m]')

    '''
    TODO:
    To create repair_duration in minutes subtract fault_bgn - fault_bgn_rounded (same for end of spell).
    Use this to establish if fault hour rounded up or down, and adjust number of repair hours based on this (x 60 minutes),
    then add back remaining start/end time.
    '''

    # reorder columns
    spells = spells[['SERNO', 'spell_bgn', 'spell_end', 'nmc_duration', 'repair_duration', 'FLT_HRS', 'PHASE',
                     'phase1', 'phase2', 'bgn_day', 'nr_faults', 'CWUC_nuniq', 'CWUC_mode', 'dom_topic_nuniq',
                     'dom_topic_mode']]

    ##### Output long form to disk ######
    summary_stats.to_csv(os.path.join(directory+'\\summary_stats', 'fault_stats.csv'), index=False)
    spells.to_csv(os.path.join(directory, 'NMC_spells.csv'), index=False)
    spells.to_csv(os.path.join(dir_shared, 'NMC_spells.csv'), index=False)
