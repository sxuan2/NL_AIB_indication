# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:13:34 2018

@author: sijian.xuan

This file is the part of "put things together"

I imported a lot of SAS datasets as csv directly, since I haven't finished 
the previous part and cannot create them in Python

All useful datasets are in datasets folder
All useful defined functions or Macros are in packages folder

In the future they might be in different folders in network drive
"""

import numpy as np
import pandas as pd
import datetime
import calendar
import os
import warnings
"""
set working directory
"""
os.chdir('/home/cdsw/packages')## adapt here
os.getcwd()

from final_indication01_macro import *
from load_model_csv import *
from proc_freq import *
from validation_b_o_rules import *
from stabilizer import *
warnings.filterwarnings('ignore')
 

"""
read files
"""
prediction_model2 = pd.read_csv("/home/cdsw/datasets/prediction_model2.csv")
prediction_model3_bbc = pd.read_csv("/home/cdsw/datasets/prediction_model3_BBC.csv")
prediction_model4_bbc = pd.read_csv("/home/cdsw/datasets/prediction_model4_BBC.csv")
input_focus = pd.read_csv("/home/cdsw/datasets/input_focus.csv")
input_comed = pd.read_csv("/home/cdsw/datasets/input_comed.csv")
knmp_1809 = pd.read_csv("/home/cdsw/datasets/knmp_1809.csv")


pat_indication_nodup = pd.read_csv("/home/cdsw/datasets/pat_indication_nodup.csv")
pat_indication_total = pd.read_csv("/home/cdsw/datasets/pat_indication_total.csv")

generic_features2 = pd.read_csv("/home/cdsw/datasets/generic_features2.csv")
indication_with_features = pd.read_csv("/home/cdsw//datasets/indication_with_features.csv")



########################## codes start here
load_model_csv("model2")
load_model_csv("model3_BBC")
load_model_csv("model4_BBC")

pat2addindication = input_focus['pat'].sort_values().unique()


#start_time = time.time()
final_indication_01 = final_indication_01(pat_indication_nodup,pat_indication_total, prediction_model2,prediction_model3_bbc,\
                        prediction_model4_bbc,pat2addindication)
#print("--- %s seconds ---" % (time.time() - start_time))

final_indication_01['indication'].value_counts()
final_indication_01['final_indication'].value_counts()

"""
also needs to fintune to ensure the final share per indication is compirable to Claims
or Sample datasets
"""
proc_freq(final_indication_01,'final_indication')
proc_freq(final_indication_01,'indication')

final_indication_02 = final_indication_01[['pat','final_indication']]
final_indication_02 = final_indication_02.rename(index = str, columns = {"final_indication":"indication"})

##create indication_with_features
temp, = np.where((generic_features2.columns.str.startswith('FL_')== True) |\
                 (generic_features2.columns.str.startswith('Focus')== True)|\
                 (generic_features2.columns == 'pat'))
temp = temp.tolist()

merge1 = pd.merge(generic_features2.iloc[:,temp],final_indication_02,how = "outer")
merge2 = pd.merge(merge1,prediction_model3_bbc,how = "outer")

indication_with_features = merge2.copy()

#start_time = time.time()
Val_01 = Validation_based_on_Rules(indication_with_features)
#print("--- %s seconds ---" % (time.time() - start_time))

proc_freq(Val_01,"same")
proc_freq_2(Val_01)

final_indication_02 = final_indication_02.rename(index = str, columns = {"indication":"indication_current"})

prevfinal = pd.read_csv("/home/cdsw/datasets/final_indication_02_08new.csv")
prevfinal = prevfinal.rename(index = str, columns = {"indication":"indication_previous"})

indication_compare = pd.merge(final_indication_02,prevfinal,how = "outer")
indication_compare.to_csv("/home/cdsw/output/indication_compare.csv")



"""
You cannot run the stabilizer. I set the path to read final_indication_02.csv in X drive. 
I already exported the files to X drive
I am not sure how to set X drive into CDSW path here.
You can read the definition in stabilizer.py in packages folder. 
"""
stabilizer04 = stabilizer("201804NEW2")
stabilizer05 = stabilizer("201805NEW2")
stabilizer06 = stabilizer("201806NEW")
stabilizer07 = stabilizer("201807NEW")
stabilizer08 = stabilizer("201808NEW")

dfs = [stabilizer04,stabilizer05,stabilizer06,stabilizer07,stabilizer08]
stabilizer_01 = pd.concat(dfs, ignore_index = True)

stabilizer_01 = stabilizer_01.sort_values(by = ['pat', 'maand'], ascending=[True, False])

### get stabilizer in line 112-123
stabilizer_02 = stabilizer_01.copy().reset_index()

## a trick to encode the nr, there is no same way of doing it as in SAS code
time = stabilizer_02['maand'].max()
first_day = datetime.date(time.year, time.month, 1)#first day of this month
days_num = calendar.monthrange(first_day.year, first_day.month)[1] #how many days in a month
first_day_of_next_month = first_day + datetime.timedelta(days = days_num) #days_num-1 is the last day of the month
first_day_of_next_month = datetime.datetime.combine(first_day_of_next_month, datetime.datetime.min.time())

## get data from most recent 12 months
stabilizer_02['nr'] =  (stabilizer_02['maand'] - first_day_of_next_month)/30 *(-1)
stabilizer_02['nr_2'] = stabilizer_02['nr'].dt.days
stabilizer_02 = stabilizer_02[stabilizer_02['nr_2'] <= 12]
"""
every time we need to check line 140-141 whether column 'nr_2' become 1,2,3,4,... no repetition just in case,
should not be a problem
"""

stabilizer_03 =stabilizer_02[['pat','indication']]
     = pd.DataFrame(stabilizer_03['pat'].groupby([stabilizer_03['indication']]).value_counts())

stabilizer_04 = stabilizer_03.copy()
stabilizer_04.columns = ['count']
stabilizer_04 = stabilizer_04.reset_index()
stabilizer_04 = stabilizer_04.rename(index = str, columns = {"indication":"indication_mode"})

#stabilizer_04.index = stabilizer_04['pat']
temp = stabilizer_04.groupby(['pat']).sum().reset_index()
temp = temp.rename(index = str, columns = {"count":"count_sum"})

stabilizer_04 = pd.merge(stabilizer_04, temp, on = 'pat')
stabilizer_04['perc'] = stabilizer_04['count']/stabilizer_04['count_sum']

stabilizer_04 = stabilizer_04.sort_values(by = ['pat', 'perc'], ascending = [True, False])
stabilizer_04.drop_duplicates(subset='pat', keep='first', inplace = True)

## get latest indication
stabilizer_05 = stabilizer_02.copy()
stabilizer_05 = stabilizer_05[['pat','indication']]
stabilizer_05 = stabilizer_05.rename(index = str, columns = {"indication":"indication_latest"})
#len(stabilizer_05['pat'].unique()) 57467
stabilizer_05.drop_duplicates(subset='pat', keep='first', inplace = True)


# check
#stabilizer_04['pat'].tolist() == stabilizer_05['pat'].tolist()

stabilizer_final = pd.merge(stabilizer_04, stabilizer_05, how = 'outer', on = 'pat')
stabilizer_final['indication'] = np.nan

cond, = np.where(stabilizer_final['perc']>0.5)
cond = cond.tolist()
stabilizer_final['indication'][cond] = stabilizer_final['indication_mode'][cond]

cond2, = np.where(stabilizer_final['perc']<= 0.5)
cond2 = cond2.tolist()
stabilizer_final['indication'][cond2] = stabilizer_final['indication_latest'][cond2]
stabilizer_final = stabilizer_final[['pat','indication']]

#final_indication_03 = stabilizer_final.copy()
stabilizer_final.to_csv("/home/cdsw/output/final_indication_"+ studyto +".(stabilized).csv")



stabilizer_final_temp = stabilizer_final.rename(index = str, columns = {"indication":"indication_current"})
prevfinal_temp = prevfinal.rename(index = str, columns = {"indication":"indication_previous"})
indication_compare2 = pd.merge(stabilizer_final_temp, prevfinal, on = 'pat')

indication_compare2.to_csv("/home/cdsw/output/indication_compare(stabilized).csv")







