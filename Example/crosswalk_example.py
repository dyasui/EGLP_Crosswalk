## A simple example demonstrating how to map 1900 population data to 2010 counties

import pandas as pd
import os
from os.path import join,split

root = split(split(__file__)[0])[0]
Crosswalks_dir = join(root,"Crosswalks")
Example_dir    = join(root,"Example")
os.chdir(Example_dir)

## reading in data

cw = pd.read_csv(join(Crosswalks_dir,'county_crosswalk_endyr_2010.csv'))

pop = pd.read_csv(join(Example_dir,"Data",'nhgis0014_ds31_1900_county.csv'))

## keeping only the relevant year, 1900
cw = cw[cw['Year'] == 1900]

## setting up the county correspondence: 
## it is "STATEA" and "COUNTYA" in pop
## and "NHGISST" and "NHGISCTY" in cw
pop['id'] = pop['STATEA'].astype(int) * 10000 + pop['COUNTYA'].astype(int)
cw['id'] = cw['NHGISST'].astype(int) * 10000 + cw['NHGISCTY'].astype(int)


pop = pop[pop['id'].isin(cw['id'])]
print(pop['AYM001'].sum())
merged = pop.merge(cw, left_on = 'id', right_on = 'id', how = 'outer')

## keeping only relevant rows:
## NHGISST_2010, NHGISCTY_2010, STATENAM_2010, NHGISNAM_2010
## AYM001 (population), weight

merged = merged[['NHGISST_2010', 'NHGISCTY_2010', 'STATENAM_2010', 'NHGISNAM_2010',
           'AYM001', 'weight']]

## re-weighting the original 1850 populations by the weights
merged['Population'] = merged['weight'] * merged['AYM001']

## aggregating to 2010 counties
output = merged.groupby(['NHGISST_2010', 'NHGISCTY_2010', 'STATENAM_2010', 'NHGISNAM_2010'])['Population'].sum().reset_index()

output.to_csv('example_output.csv', index = False)
print(output['Population'].sum())

#--- Uncomment to auto-open ---
# os.system('example_output.csv')
