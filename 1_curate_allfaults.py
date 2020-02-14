"""
Author: Joe King
Date: 5 Feb. 2020

    Script cleans and manipulates original fault data (blackhawks only) and outputs to CSV. Original files are
    saved separately by calendar year, a format this script preserves.

    However, '2017-2019 new fault data.csv' comprises multiple years of newer fault data, which was combined
    in the script '0_append_newfault_data.py'.
    
"""

import pandas as pd
import numpy as np
import os, re, warnings
pd.options.display.max_rows = 150
pd.options.display.max_columns = 75
pd.options.mode.chained_assignment = None # turns of SettingWithCopyWarning
warnings.simplefilter(action='ignore',category=FutureWarning)


def main(input_dir, dataframe, min_date_cutoff=None, max_date_cutoff=None):
    '''
    Function cleans a given dataframe and outputs to CSV file. This includes:
        1) dropping sparse or non-relevent seeming columns, and renaming columns; 
        2) dropping duplicate records on 'KEY13';
        3) formatting dates;
        4) dropping records with missing start and/or end date;
        5) dropping records whose start date was before the minimum and/or maximum date cut-off;
        6) dropping records where end date before start date;
        7) dropping records missing serial number;
        8) misc formatting to variables;
        9) creating a record status variable for the status associated with that fault record;
    
    :input_dir: str, path to original data
    :dataframe: str, name of CSV to process
    :min_date_cutoff: YYYY-MM-DD date to exclude faults before
    :max_date_cutoff: YYYY-MM-DD date to exclude faults after
        
    '''
    print("\nBeginning curation of '{}'.".format(dataframe))
    df = pd.read_csv(os.path.join(input_dir, dataframe), index_col=None)
    
    ####################################################################
    ##### RENAMING, DROPPING USELESS COLS & ROWS, FORMATTING DATES #####
    ####################################################################
    
    # drop Pandas generated index column
    if "Unnamed: 0" in df.columns:
        del df['Unnamed: 0']
    
    # Strip whitespace, if any from strings
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    
    # Remove annoying < > ? in column names
    r = re.compile('(<|>|\?)')
    df.columns = [r.sub('', i) for i in df.columns]
    

    # verify no duplicate records
    before = df.shape[0]
    df.drop_duplicates(subset='KEY13', keep='first', inplace=True)
    after = df.shape[0]
    share = (1-(after/before))*100
    print(share, "percent rows duplicated and dropped from", dataframe)
    del before, after, share
    
    # drop nearly completely empty columns
    for x in df.columns:
        if df[x].isnull().mean() >= 0.99:
            print("Column", x, "is nearly completely missing and was deleted")
            del df[x]
        else:
            pass
    del x
    
    # Drop other misc columns that not helpful
    del_cols = ['FNO', 'DISC_PID', 'EI_ID', 'CLOSED', 'CNAME', 'RFG', 'IN HOLDING AREA']
    df.drop(columns=[i for i in df.columns if i in del_cols], inplace=True); del del_cols
    df.drop(columns=[i for i in df.columns if "SCD" in i], inplace=True)
    
    # FDATE and CDATE formatted differently across airframes
    for x in ['FDATE', 'CDATE']:
        test = pd.to_datetime(df[x], format ='%d-%b-%Y', errors='coerce')  
        if test.isnull().mean() == 1:
            test = pd.to_datetime(df[x], format ='%d %b %Y', errors='coerce')
            if test.isnull().mean() == 1:
                test = pd.to_datetime(df[x], format ='%d-%b-%y', errors='coerce')
                if test.isnull().mean() == 1:
                    raise ValueError("\nDate formatting of FDATE and/or CDATE incorrect")
                else:
                    foo = test # for '%d-%b-%y' format
            else:
                foo = test # for '%d %b %Y' format
        else:
            foo = test # for '%d-%b-%Y' format

        # 2017-2019 new files have different formatting
        df.loc[foo.notnull(), x + '_mod'] = foo
        df.loc[foo.isnull(), x + '_mod'] = pd.to_datetime(df[x], format ='%Y-%m-%d %H:%M', errors='coerce')
        df[x] = df[x + '_mod']
        del df[x + '_mod']
    

    # Check share records missing start and/or end date. Delete those missing start date  
    print("\n\n", df['CDATE'].isnull().mean().round(4), 'proportion records missing CDATE')
    print(df['FDATE'].isnull().mean().round(4), 'proportion records missing FDATE and deleted')
    print(df['SERNO'].isnull().mean().round(4), 'proportion records missing SERNO and deleted')
    df = df.loc[(df.FDATE.notnull()) & (df.CDATE.notnull())] # delete records missing start and/or end date
    df = df.loc[df.FDATE <= df.CDATE]  # if end date before start date, drop. Ignore time inconsistencies for now
    df = df.loc[df['SERNO'].notnull()] # delete records missing serial number
    
    # Delete records outside desired range
    print("\nEarliest FDATE: '{}'.".format(df['FDATE'].min()))
    if min_date_cutoff is not None:
        df = df.loc[df['FDATE'] >= min_date_cutoff] # deletes records prior to cutoff
        print("\nAfter restrictions earliest FDATE: '{}'.".format(min(df['FDATE'].dt.date)))
    print("\nMost recent FDATE: '{}'.".format(max(df['FDATE'].dt.date)))
    if max_date_cutoff is not None:
        df = df.loc[(df['FDATE'] <= max_date_cutoff)] # deletes records after to cutoff
    df.sort_values(by='KEY13', inplace=True) # sorts descending by model, serial number, date, fault number
    df = df.reset_index(drop=True)
    
    # Random fixes
    df['SERNO'].replace('\'', '', regex=True, inplace=True)
    df['UNIT'].replace('Dont Know', np.NaN, inplace=True) # not a valid unit name
    df.loc[(df['TMMH'] < 0) | (df['TMMH'] > 1000), 'TMMH'] = np.NaN # total man hours can't be thousands or negative
    df['TYPE'].replace('.', np.NaN, inplace=True)
    df['CWUC'] = df['CWUC'].str.replace('\'', '')
    df['CWUC'] = df['CWUC'].str[:2].replace(['AA', '98', '0'], '00')  # part of aircraft a fault applies to
    df['CWUC'] = pd.to_numeric(df['CWUC'], errors='coerce')
    df['CWUC'].fillna(0, inplace=True)

    for col in ['ACHRS', 'CACHRS']:
        df[col] = df[col].round(0) # round to remove potential human error
    df.loc[(df['ACHRS'].isnull()) & (df['CACHRS'].notnull()), 'ACHRS'] = df['CACHRS'] # CACHRS (end hrs) more often missing, fill with ACHRS (begin hrs)
    df.rename(columns={'ACHRS': 'FLT_HRS'}, inplace=True); del df['CACHRS']

    ####################################################################
    #####     CREATE FAULT VARIABLE & OUTPUT ALL FAULTS DATASET    #####
    ####################################################################
    
    # how each record affects flight status of bird (repairs may be overlapping in time)
    df['record_status'] = np.NaN
    df.loc[df['STAT'] == 'X', 'record_status'] = 'NMC' # Red X, this is clear no fly
    df.loc[df['STAT'] == '/', 'record_status'] = 'PMC' # fix is needed at some point, but not urgent or dangerous to ground plane
    df.loc[df['STAT'] == '-', 'record_status'] = 'PPM' # most frequent, looks to be going thru maintenance process, actual status may change
    df.loc[df['STAT'].isin(['N', 'B', 'C', 'R']), 'record_status'] = 'PPM' # I'm guessing
    df.loc[df['STAT'] == '+', 'record_status'] = 'PMC' # Circle red X (conditionally flyable) acc'd codebook. Confirmed by Scott Moyer
    df = df.loc[df['record_status'].notnull()] # drop missing
    
    print("\nFreq dist of record statuses \n\n", df.record_status.value_counts(dropna=False, normalize=True))
    
    
    ####################################################################
    #####    OUTPUT CLEANED INTERMEDIATE ALL FAULTS DATASET        #####
    ####################################################################

    # Where output final product
    output_dir = os.path.join(r'X:\Pechacek\ARNG Aircraft Readiness\Data\Processed\Master fault data intermediate', os.path.split(input_dir)[-1])
    output_dir_local = "../data" # save local copy to run faster
    
    # Output intermediate dataset with all faults
    print("\nOutputting all faults of {}".format(dataframe))
    df.to_csv(os.path.join(output_dir, dataframe[:-4] + '_all_faults' + '.csv'), index=False)
    df.to_csv(os.path.join(output_dir_local, dataframe[:-4] + '_all_faults' + '.csv'), index=False)
    print("Done")


if __name__ == "__main__":
    
    
    min_date_cutoff = '2010-10-15'
    max_date_cutoff = '2019-09-15'

    input_dir = 'X:/Pechacek/ARNG Aircraft Readiness/Data/Processed/Master fault data unduped/NGB Blackhawk'
    
    files = [i for i in os.listdir(input_dir) if i.lower().endswith('.csv')]
    files.sort()
    
    for file in files[10:19]: # restricts to CY10-18
        try:
            main(input_dir, file, min_date_cutoff, max_date_cutoff)
        except:
            print("\nI'm having trouble reading file {}".format(file))

    # New 2017-19 fault data
    path = r'\\div-sfrd.sfrd.ida.org\public\Pechacek\ARNG Aircraft Readiness\Data\Processed\appended new 2017-2017 fault data from Scott Moyers'
    main(input_dir=path, dataframe='2017-2019 new fault data.csv', min_date_cutoff='2010-10-15', max_date_cutoff='2019-09-15')