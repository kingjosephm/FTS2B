import pandas as pd
import os
pd.options.display.max_rows = 150
pd.options.display.max_columns = 75

dir1 = r'\\div-sfrd.sfrd.ida.org\public\Pechacek\ARNG Aircraft Readiness\Data\Original\20191213 FY18-FY19 fault data from Scott Moyers'
df1 = pd.read_excel(os.path.join(dir1, 'Blackhawk data for IDA 1.xlsx'))
df2 = pd.read_excel(os.path.join(dir1, 'Blackhawk data for IDA 2.xlsx'))

dir2 = r'\\div-sfrd.sfrd.ida.org\public\Pechacek\ARNG Aircraft Readiness\Data\Original\20191216 FY17 fault data from Scott Moyers'
df3 = pd.read_csv(os.path.join(dir2, 'Blackhawk 2_1 data for IDA.csv'))
df4 = pd.read_csv(os.path.join(dir2, 'Blackhawk 2_2 dat for IDA.csv'))

dir3 = r'\\div-sfrd.sfrd.ida.org\public\Pechacek\ARNG Aircraft Readiness\Data\Original\20200123 Additional fault data from Scott Moyers'
df5 = pd.read_csv(os.path.join(dir3, 'Blackhawk 2_1 data for IDA.csv'))
df6 = pd.read_csv(os.path.join(dir3, 'Blackhawk 2_2 dat for IDA.csv'))

####### APPEND ######

df = pd.DataFrame()
df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

df = df.sort_values(by=['KEY13', 'SOURCE']).reset_index(drop=True)

del df['SOURCE']

undup = df.drop_duplicates(keep='first').reset_index(drop=True)

path = r'\\div-sfrd.sfrd.ida.org\public\Pechacek\ARNG Aircraft Readiness\Data\Processed\appended new 2017-2017 fault data from Scott Moyers'
undup.to_csv(os.path.join(path, '2017-2019 new fault data.csv'), index=False)

