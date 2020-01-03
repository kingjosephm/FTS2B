# -*- coding: utf-8 -*-
"""
Author: Joe King
Date: 10 Sept. 2019

    Script takes all corrected faults for blackhawks, keeps only NMC faults
    and whose system flag indicate that fault was on the airframe itself.
    Appends each calendar year's faults at end and outputs to CSV.
    
"""

import pandas as pd
import numpy as np
import os, warnings, re
pd.options.display.max_rows = 150
pd.options.display.max_columns = 75
pd.options.mode.chained_assignment = None # turns of SettingWithCopyWarning
warnings.simplefilter(action='ignore',category=FutureWarning)

def unify_flt_hrs(df):
    '''
    Function takes fault start flight hours, and fills with fault end flight hours if the former is missing and latter is not.
    Renames column and outputs dataframe with new flight hours column.
    :df: Pandas df
    '''
    for col in ['ACHRS', 'CACHRS']:
        df[col] = df[col].round(0) # round to remove potential human error    
    df.loc[(df['ACHRS'].isnull()) & (df['CACHRS'].notnull()), 'ACHRS'] = df['CACHRS'] # CACHRS (end hrs) more often missing, fill with ACHRS (begin hrs)
    df.rename(columns={'ACHRS': 'FLT_HRS'}, inplace=True); del df['CACHRS']
    return df

def get_nmc_faults(df):
    '''
    Function takes dataframe of all faults, restricts to NMC faults, creates 
        a corrected flight hours variable, and outputs these NMC faults only.
        
    :df: pandas dataframe
    '''
    # NMC records only, SYS == A, non-missing flight hours
    nmc = df.loc[(df['record_status'] == 'NMC') & (df['SYS'] == 'A')]
    nmc.drop(columns=['SYS', 'record_status'], inplace=True)
    nmc = nmc[['SERNO', 'FDATE', 'FTIME', 'CDATE', 'CTIME', 'FLT_HRS', 'IN_PHASE', 'FAULT', 'ACTION', 'CWUC']]
    nmc.rename(columns={'IN_PHASE': 'PHASE'}, inplace=True)
    nmc['PHASE'] = np.where(nmc['PHASE']=='X', 1, 0) # Recode in phase to numeric
    nmc.sort_values(by=['SERNO', 'FDATE', 'FTIME'], inplace=True)
    return nmc

def phase_flags(pattern):
    '''
    :param pattern: Regex pattern (as string) to create dummy variable for
    :return: Pandas Series (concatenated to df) as dummy (True == 1)
    '''
    column = df['FAULT'].str.contains(pattern)
    return np.where(column==True, 1, 0) # convert to dummy for later groupby operation

if __name__ == '__main__':
    

    directory = r'X:\Pechacek\ARNG Aircraft Readiness\Data\Processed\Master fault data intermediate\NGB Blackhawk'
    
    files = [i for i in os.listdir(directory) if i.endswith('.csv') and "all_faults" in i]
    files.sort()
    
    # Append each year's Blackhawk NMC spells together
    cols = ['SERNO', 'FDATE', 'FTIME', 'CDATE', 'CTIME', 'SYS', 
            'record_status', 'ACHRS', 'CACHRS', 'IN_PHASE', 
            'FAULT', 'ACTION', 'CWUC']
    
    dfs = []
    processed_ct = pd.DataFrame(data={'dataframe': [], 'drop missing SERNO, FDATE, CDATE, flt hrs, all dups':[]})
    nmc_ct = pd.DataFrame(data={'dataframe': [], 'NMC faults':[]})
    for file in files:
        print(file)
        df = pd.read_csv(os.path.join(directory, file), usecols=cols)
        df = unify_flt_hrs(df)
        df['CWUC'] = df['CWUC'].str[:2].replace(['AA', '98', '0'], '00') # part of aircraft a fault applies to
        df['CWUC'] = pd.to_numeric(df['CWUC'], errors='coerce')
        df['CWUC'].fillna(0, inplace=True)
        df = df.loc[(df.FDATE.notnull()) & (df.CDATE.notnull()) & (df.FLT_HRS.notnull()) & (df.SERNO.notnull())]
        df = df.loc[df.FDATE <= df.CDATE] # if end date before start date, drop. Ignore time inconsistencies for now
        df = df.drop_duplicates()
        processed_ct = processed_ct.append({'dataframe': file[:19], 'drop missing SERNO, FDATE, CDATE, flt hrs, all dups':len(df)}, ignore_index=True)
        df = get_nmc_faults(df)
        nmc_ct = nmc_ct.append({'dataframe':file[:19], 'NMC faults': int(len(df))}, ignore_index=True)
        dfs.append(df)
    df = pd.concat(dfs); del dfs, file, files
    df.sort_values(by=['SERNO', 'FDATE', 'FLT_HRS'], inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)

    # Create new phase flag using fault text string
    df['phase_fault'] = phase_flags('PMI|PHASE') # note - in script '4_nmc_spells' union of this and PHASE variable is obtained and used
    df['phase1'] = phase_flags('PMI1|PMI 1|PMI \#1|PMI\#1|PMI-1|PMI - 1|PMII|PMI I|PMI-I|PMI - I|360HOUR|360 HOUR|360HR|360 HR|480HOUR|480 HOUR|480HR|480 HR')
    df['phase2'] = phase_flags('PMI2|PMI 2|PMI \#2|PMI\#1|PMI-2|PMI - 2|PMIII|PMI II|PMI-II|PMI - II|720HOUR|720 HOUR|720HR|720 HR|960HOUR|960 HOUR|960HR|960 HR')

    # Output appended NMC faults
    df.to_csv(os.path.join(directory, 'NMC_faults.csv'), index=False)
    
    # Output fault counts
    summary_stats = pd.read_csv(os.path.join(directory+'\\summary_stats', 'Blackhawk fault counts.csv'), 
                                usecols=['dataframe', 'all_faults'])
    summary_stats = summary_stats.merge(processed_ct, on='dataframe')
    summary_stats = summary_stats.merge(nmc_ct, on='dataframe')
    summary_stats.to_csv(os.path.join(directory+'\\summary_stats', 'Blackhawk fault counts.csv'), index=False)