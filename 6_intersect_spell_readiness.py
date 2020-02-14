# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:52:14 2019

    Script enforces the intersection between readiness months (serial number x month) and fault spells,
    to ensure the same population between readiness and fault data.

@author: jking
"""


import pandas as pd
import os
pd.options.display.max_rows = 200
pd.options.display.max_columns = 75
pd.options.mode.chained_assignment = None # turns of SettingWithCopyWarning


if __name__ == '__main__':

    
    ##################################################
    #####           READ AND PROCESS             #####
    ##################################################
    
    # Fault Data
    try:
        fault_dir = r'C:\Users\jking\Documents\FTS2B\data'
    except:
        fault_dir = r'X:\Pechacek\ARNG Aircraft Readiness\Data\Processed\Master fault data intermediate\NGB Blackhawk'
    f = pd.read_csv(os.path.join(fault_dir, 'NMC_spells.csv'))
    f.rename(columns={'SERNO':'serial_number'}, inplace=True)
    serno = f.serial_number.unique().tolist() # blackhawk serial numbers in fault data, for subsetting below   
    
    # Readiness Data
    readiness_dir = r'X:\Pechacek\ARNG Aircraft Readiness\Data\Processed\Tables for merge'
    r = pd.read_csv(os.path.join(readiness_dir, 'readiness_covariates_201011_201909.csv'),
                            encoding='ISO-8859-1', low_memory=False); del readiness_dir

    r = r.loc[r.serial_number.isin(serno) & (r.report_date >='2010-10-01'), 
                    ['serial_number', 'report_date', 'facility_id']]; del serno
    r['serial_number'] = r['serial_number'].astype(int)
    r = r.loc[r.facility_id.notnull()] # drop any serial numbers missing facility ID
    r.sort_values(by=['serial_number', 'report_date'], inplace=True)
    r.reset_index(drop=True, inplace=True)

    #################################################################
    ##### Convert to datetime, create readiness month for merge #####
    #################################################################

    f['spell_bgn'] = pd.to_datetime(f['spell_bgn'], format ='%Y-%m-%d %H:%M', errors='raise')
    f['spell_end'] = pd.to_datetime(f['spell_end'], format ='%Y-%m-%d %H:%M', errors='raise')

    f.loc[f.spell_bgn.dt.day <= 15, 'read_startmo'] = \
        (f['spell_bgn']).dt.strftime('%Y-%m') + '-15'
    f.loc[f.spell_bgn.dt.day > 15, 'read_startmo'] = \
        (f['spell_bgn'] + pd.DateOffset(months=1)).dt.strftime('%Y-%m') + '-15'
    # Spell end readiness month
    f.loc[f.spell_end.dt.day <= 15, 'read_endmo'] = \
        (f['spell_end']).dt.strftime('%Y-%m') + '-15'
    f.loc[f.spell_end.dt.day > 15, 'read_endmo'] = \
        (f['spell_end'] + pd.DateOffset(months=1)).dt.strftime('%Y-%m') + '-15'

    f['read_startmo'] = pd.to_datetime(f['read_startmo'], format ='%Y-%m-%d', errors='raise')
    f['read_endmo'] = pd.to_datetime(f['read_endmo'], format ='%Y-%m-%d', errors='raise')
    r['report_date'] = pd.to_datetime(r['report_date'], format ='%Y-%m-%d', errors='raise')
    
    # Roll back readiness month of spell by 1 to ensure chopper at same facility_id throughout whole spell
    f['read_startmo'] = (f['read_startmo'] + pd.DateOffset(months=-1))
    
    # Serial number x spell number, uniquely identifies spell
    f['spell_num'] = f.groupby('serial_number').cumcount()
    uniq_spells_orig = f.shape[0] # number of unique spells before merge
    
    ##################################################
    #####   INTERSECTION WITH READINESS DATA     #####
    ##################################################
    
    # Subset to relevant columns
    foo = f[['serial_number', 'read_startmo', 'read_endmo', 'spell_num']]
    
    # Convert wide to long for read_startmo and read_endmo, ignoring duplicate start and/or end months
    long = pd.melt(foo, id_vars=['serial_number', 'spell_num'], value_name='report_date') # report_date is either start or end month
    long = long.sort_values(by=['serial_number', 'spell_num', 'report_date']).reset_index(drop=True) # sort for forward fill
    
    # Wide to long conversion
    print("\nConverting wide to long..")
    mo = long.set_index('report_date').groupby(['serial_number', 'spell_num']).resample('M').ffill()
    mo = mo[[]].reset_index(drop=False) 
    mo['report_date'] = mo['report_date'].dt.strftime('%Y-%m') + '-15' # pandas annoyingly sets to the end of the month, this resets back to 15th
    mo['report_date'] = pd.to_datetime(mo['report_date'], format ='%Y-%m-%d', errors='raise')
    del foo, long
    
    # Full outer join with readiness data
    mo['nr_months'] = mo.groupby(['serial_number', 'spell_num']).transform('count') # number of readiness months per spell, should be same after intersection else drop (below)
    uniq_fault_months_orig = mo.drop_duplicates(subset=['serial_number', 'report_date']).shape[0] # total unique fault months before merge
    join = r.merge(mo, on=['serial_number', 'report_date'], how='outer', indicator=True) # retain duplicates in merge to link back to original spells
    join = join.sort_values(by=['serial_number','spell_num', 'report_date']).reset_index(drop=True)
    
    
    # Summary stats of merge
    stats = join.drop_duplicates(subset=['serial_number', 'report_date'])['_merge'].value_counts()\
        .rename({'both':'intersection', 'right_only':'fault_only', 'left_only':'readiness_only'})\
        .reset_index().rename(columns={'index':'merge_type', '_merge':'n_months'})
    stats['share_merge'] = stats['n_months'] / len(join.drop_duplicates(subset=['serial_number', 'report_date']))
    print(stats)
    
    # Keep intersection
    inters = join.loc[join._merge == 'both'].sort_values(by=['serial_number', 'spell_num', 'report_date']).reset_index(drop=True).drop(columns=['_merge'])
    uniq_fault_months_inters = len(inters.drop_duplicates(subset=['serial_number', 'report_date'])) # total unique fault months in intersection
    
    # Keep only spells where total number of spell months (orig) equals total number spell months after intersection
    inters['post'] = inters.groupby(['serial_number', 'spell_num'])['serial_number'].transform('count')
    inters = inters.loc[inters['post'] == inters['nr_months']]
    inters.drop(columns=['post', 'nr_months'], inplace=True)
    uniq_fault_months_drop_part_spells = inters.drop_duplicates(subset=['serial_number', 'report_date']).shape[0] # unique fault months 
    
    
    # What proportion of spell-months are reported a different facility_id?
    inters['uniq_facid'] = inters.groupby(['serial_number', 'spell_num'])['facility_id'].transform('nunique')
    uniq_facids = inters.drop_duplicates(subset=['serial_number', 'spell_num'])['uniq_facid'].value_counts().reset_index().rename(columns={'index':'uniq_facil_count', 'uniq_facid':'n_spells'}) # uniq facilities grouped by serial number & spell
    uniq_facids['share'] = uniq_facids['n_spells'] / len(inters.drop_duplicates(subset=['serial_number', 'spell_num']))
    
    # Drop all spell-months (and thereby spells in subsequent merge) if changed facility_id during spell
    inters = inters.loc[inters['uniq_facid'] == 1]
    uniq_fault_months_facid = inters.drop_duplicates(subset=['serial_number', 'report_date']).shape[0]
    
    
    # Merge back with spell data
    inters = inters[['serial_number', 'spell_num']].drop_duplicates()
    f = f.merge(inters, on=['serial_number', 'spell_num'], how='inner')
    final_spells = len(f)
    
    del f['spell_num']
    
    # Roll forward readiness month of spell back to where was originally
    f['read_startmo'] = (f['read_startmo'] + pd.DateOffset(months=1))
    
    print("\nDone processing, outputting file to disk")
    final_dir = r'\\div-sfrd.sfrd.ida.org\public\Pechacek\ARNG Aircraft Readiness\Data\Processed\Tables for merge'
    f.to_csv(os.path.join(final_dir, 'NMC_spells_final.csv'), index=False)
    f.to_csv(os.path.join(fault_dir, 'NMC_spells_final.csv'), index=False) # put copy in intermediate directory
    
    
    ##################################################
    #####       SUMMARY STATS OF MERGES          #####
    ##################################################
    
    # Combine various counts
    df = pd.DataFrame()
    df = pd.DataFrame([uniq_spells_orig, uniq_fault_months_orig, uniq_fault_months_inters,
                       uniq_fault_months_drop_part_spells, uniq_fault_months_facid, final_spells]).rename({0:'orig spells', 1:'spell months', 2:'spell months intersection',
                        3:'spell months drop part spells', 4:'spell months constant facility_id',
                        5:'final spells'}).reset_index().rename(columns={'index':'explanation', 0:'n_unique'})
    print(df)
    
    # Output merge stats to Excel workbook
    writer = pd.ExcelWriter(os.path.join(fault_dir+'\\summary_stats', 'readiness_fault_merge_stats.xlsx'), engine='openpyxl')
    stats.to_excel(writer, sheet_name='merge_month', index=False)
    df.to_excel(writer, sheet_name='merge_counts', index=False)
    uniq_facids.to_excel(writer, sheet_name='facility_counts', index=False)
    writer.save()