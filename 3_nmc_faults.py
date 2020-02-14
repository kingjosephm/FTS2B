# -*- coding: utf-8 -*-
"""
Author: Joe King
Date: 5 Feb. 2020

    Script takes all corrected faults for blackhawks, keeps only NMC faults
    and whose system flag indicate that fault was on the airframe itself.
    Appends each calendar year's faults at end and outputs to CSV.
    
"""

import pandas as pd
import numpy as np
import os, warnings
pd.options.display.max_rows = 150
pd.options.display.max_columns = 75
pd.options.mode.chained_assignment = None # turns of SettingWithCopyWarning
warnings.simplefilter(action='ignore',category=FutureWarning)


def get_nmc_faults(df):
    '''
    Function takes dataframe of all faults, restricts to NMC faults, creates 
        a corrected flight hours variable, and outputs these NMC faults only.
        
    :df: pandas dataframe
    '''
    # NMC records only, SYS == A
    nmc = df.loc[(df['record_status'] == 'NMC') & (df['SYS'] == 'A')]
    nmc.drop(columns=['SYS', 'record_status'], inplace=True)
    nmc = nmc[['SERNO', 'FDATE', 'FTIME', 'CDATE', 'CTIME', 'FLT_HRS', 'PHASE', 'FAULT', 'ACTION', 'CWUC']]
    nmc.sort_values(by=['SERNO', 'FDATE', 'FTIME'], inplace=True)
    return nmc

def create_flags(pattern):
    '''
    :param pattern: Regex pattern (as string) to create dummy variable for, seaching across 'FAULT' and 'ACTION' fields.
    :return: Pandas Series (concatenated to df) as dummy (True == 1)
    '''
    column = (df['FAULT'] + ' ' + df['ACTION']).str.contains(pattern)
    return np.where(column==True, 1, 0) # convert to dummy for later groupby operation

if __name__ == '__main__':

    directory = '../data'
    dir_shared = r'X:\Pechacek\ARNG Aircraft Readiness\Data\Processed\Master fault data intermediate\NGB Blackhawk'
    
    df = pd.read_csv(os.path.join(directory, 'all_faults_appended.csv'))

    df = get_nmc_faults(df)

    # Create new phase flag using fault text string
    df['phase_fault'] = create_flags('PMI|PHASE') # note - in script '4_nmc_spells' union of this and PHASE variable is obtained and used
    df['phase1'] = create_flags('PMI1|PMI 1|PMI \#1|PMI\#1|PMI-1|PMI - 1|PMII|PMI I|PMI-I|PMI - I|360HOUR|360 HOUR|360HR|360 HR|480HOUR|480 HOUR|480HR|480 HR')
    df['phase2'] = create_flags('PMI2|PMI 2|PMI \#2|PMI\#1|PMI-2|PMI - 2|PMIII|PMI II|PMI-II|PMI - II|720HOUR|720 HOUR|720HR|720 HR|960HOUR|960 HOUR|960HR|960 HR')

    # Output appended NMC faults
    df.to_csv(os.path.join(directory, 'NMC_faults.csv'), index=False)
    df.to_csv(os.path.join(dir_shared, 'NMC_faults.csv'), index=False)