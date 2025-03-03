import pandas as pd
import geopandas as gpd
import os
from os.path import join, split
import time
import numpy as np
from zipfile import ZipFile
import tempfile
import re

# Path setup
path = split(__file__)[0]
root = split(path)[0]
os.chdir(path)

# Setting up the dataframe
master_df = gpd.GeoDataFrame()

def append_0(string):
    return string + '0'

cw = pd.read_csv('Code/state_name_cw.csv')
states = pd.read_csv('Code/states_union.csv')

def fix_2010(shp, cw=cw):
    shp = shp[['STATEFP10', 'COUNTYFP10', 'NAME10', 'geometry']]
    shp = shp.rename(
        columns = {
            'STATEFP10': 'NHGISST',
            'COUNTYFP10': 'NHGISCTY',
            'NAME10': 'NHGISNAM'
        }
    )
    shp['NHGISST'] = shp['NHGISST'].apply(append_0).astype(int)
    shp = shp.merge(cw, on='NHGISST')
    shp['ICPSRCTY'] = np.nan
    shp['ICPSRST'] = np.nan
    return shp

end_year = '1950'
other_years = [
    # '1790','1800','1810','1820','1830','1840','1850','1860','1870','1880', '1890',
    '1900','1910','1920','1930','1940','1950',
    # '1960','1970','1980', '1990','2000','2010'
]
other_years.remove(end_year)


# --- 1) Find the most recent NHGIS shape zip in Shapefiles/ (highest number) ---
shapefiles_dir = join(root, "Shapefiles")
all_files = os.listdir(shapefiles_dir)

# Collect files that match the pattern "nhgisXXXX_shape.zip"
candidate_files = []
pattern = r'^nhgis(\d+)_shape\.zip$'
for f in all_files:
    if re.match(pattern, f):
        candidate_files.append(f)

if not candidate_files:
    raise FileNotFoundError(
        f"No file matching pattern 'nhgis****_shape.zip' found in {shapefiles_dir}"
    )

# Sort by the numeric portion, pick the highest
candidate_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
latest_zip_filename = candidate_files[-1]
big_zip_path = join(shapefiles_dir, latest_zip_filename)

# Extract the same numeric portion for nested paths
match = re.match(pattern, latest_zip_filename)
if not match:
    raise RuntimeError(
        f"Could not parse numeric portion from '{latest_zip_filename}' using pattern {pattern}"
    )
prefix_number = match.group(1)  # e.g. "0032"
prefix = f"nhgis{prefix_number}_shape"  # e.g. "nhgis0032_shape"


# --- 2) Helper to read a shapefile for a given year from the nested zip ---
def read_shapefile_from_nested_zip(big_zip_path, year):
    """
    Opens the main 'nhgis****_shape.zip', locates the nested zip
    'nhgis****_shape/nhgis****_shapefile_tl2008_us_county_{year}.zip',
    extracts that zip's shapefile to a temp dir, and returns a GeoDataFrame.
    """
    nested_zip_name = f"{prefix}/{prefix}file_tl2008_us_county_{year}.zip"
    with ZipFile(big_zip_path, "r") as main_zip:
        with main_zip.open(nested_zip_name) as nested_zip_file:
            with ZipFile(nested_zip_file) as year_zip:
                with tempfile.TemporaryDirectory() as tmpdir:
                    year_zip.extractall(tmpdir)
                    shp_path = os.path.join(tmpdir, f"US_county_{year}_conflated.shp")
                    return gpd.read_file(shp_path)

# --- 3) Read the "end_year" shapefile from the nested zip ---
shp_end = read_shapefile_from_nested_zip(big_zip_path, end_year)
if end_year == '2010':
    shp_end = fix_2010(shp_end)

# Rename columns in the end-year shapefile
cols = shp_end.columns
new_cols_end = []
for col in cols:
    if col != 'geometry':
        new_cols_end.append(col + '_' + end_year)
    else:
        new_cols_end.append(col)
shp_end.columns = new_cols_end

# --- 4) Loop over OTHER years, overlay & accumulate crosswalk info ---
for year in other_years:
    start = time.time()

    shp = read_shapefile_from_nested_zip(big_zip_path, year)
    if year == '2010':
        shp = fix_2010(shp)

    shp['Year'] = year
    shp['area_base'] = shp.area

    # Intersect with the end_year shapefile
    temp = gpd.overlay(shp, shp_end, how='intersection')
    
    # Compute weights
    temp['area'] = temp.area
    temp['weight'] = temp['area'] / temp['area_base']

    # Keep only relevant columns
    temp = temp[
        [
            'Year','NHGISST','NHGISCTY','STATENAM','NHGISNAM','ICPSRST','ICPSRCTY',
            'area_base',
            'NHGISST_'+end_year, 'NHGISCTY_'+end_year,
            'STATENAM_'+end_year, 'NHGISNAM_'+end_year,
            'ICPSRST_'+end_year, 'ICPSRCTY_'+end_year,
            'area','weight'
        ]
    ]
    
    # Drop very tiny polygons
    temp = temp[temp['area'] > 10]
    
    # Renormalize weights
    reweight = temp.groupby(['NHGISCTY','NHGISST'])['weight'].sum().reset_index()
    reweight['new_weight'] = reweight['weight']
    reweight = reweight.drop('weight', axis=1)

    temp = temp.merge(reweight, on=['NHGISCTY','NHGISST'])
    temp['weight'] = temp['weight'] / temp['new_weight']
    temp = temp.drop('new_weight', axis=1)
    
    # Mark if state was in the union that year
    states_year = states[states[year] == 1]['State']
    temp['US_STATE'] = 0
    temp.loc[temp['STATENAM'].isin(states_year.apply(str.strip)), 'US_STATE'] = 1

    # Append to master
    master_df = master_df.append(temp)
    print(year, time.time() - start, "seconds")


# --- 5) Save the combined crosswalk ---
output_filename = 'county_crosswalk_endyr_' + end_year + '.csv'
master_df.to_csv(output_filename, index=False)
print("Done! Wrote", output_filename)
