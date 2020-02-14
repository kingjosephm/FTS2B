
'''

Author: Joe King
Date: 5 Feb. 2020

    Script appends single (calendar) year of faults - all faults - unduplicates faults, and keeps a subset of
    columns. Saves this dataframe as one large CSV

'''

import pandas as pd
import numpy as np
import os, warnings
pd.options.display.max_rows = 150
pd.options.display.max_columns = 75
pd.options.mode.chained_assignment = None # turns of SettingWithCopyWarning
warnings.simplefilter(action='ignore',category=FutureWarning)

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

    files = [i for i in os.listdir(directory) if i.endswith('.csv') and "all_faults" in i]
    files.sort()

    # Subset of columns for subsequent analysis
    cols = ['SERNO', 'FDATE', 'FTIME', 'CDATE', 'CTIME', 'SYS',
            'record_status', 'FLT_HRS', 'IN_PHASE',
            'FAULT', 'ACTION', 'CWUC']

    dfs = []
    for file in files:
        print(file)
        df = pd.read_csv(os.path.join(directory, file), usecols=cols)
        dfs.append(df)
    df = pd.concat(dfs)
    del dfs, file, files
    df.sort_values(by=['SERNO', 'FDATE', 'CDATE', 'FLT_HRS'], inplace=True)
    print(df.duplicated().value_counts(normalize=True))
    df = df.drop_duplicates().reset_index(drop=True)  # drops duplicate faults across files

    # Flags for deployment
    df['deploy'] = create_flags('Missile Warning System|MWS|MWO|AAR-57|AN/AAR-57|AN-57|Ballistic Protection Systems|BPS|COUNTER MEASURES|AVR-2B|AWR')

    df['deploy_inst'] = create_flags(' INSTALL|APPLICATION|REINSTALL|RETROFIT|COMPLETED')
    df['deploy_remov'] = create_flags('REMOV|UNINSTALL|TERMINAT')
    for _ in ['deploy_inst', 'deploy_remov']:
        df[_] = np.where((df[_]==1) & (df['deploy']==0), 0, df[_]) # code to 0 (n/a) if no deploy flag
    df['deploy_other'] = np.where((df.deploy==1) & (df.deploy_inst==0) & (df.deploy_remov==0), 1, 0)

    # order columns
    df = df[cols+['deploy', 'deploy_inst', 'deploy_remov', 'deploy_other']]

    df.rename(columns={'IN_PHASE': 'PHASE'}, inplace=True)
    df['PHASE'] = np.where(df['PHASE']=='X', 1, 0) # Recode in phase to numeric

    # output
    df.to_csv(os.path.join(directory, 'all_faults_appended.csv'), index=False)
    df.to_csv(os.path.join(dir_shared, 'all_faults_appended.csv'), index=False)