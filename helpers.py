import pandas as pd


df = pd.read_csv('./data_set.csv')

df = df[['building_class', 'facility_type', 'energy_star_rating', 'year_built']]

building_class = df['building_class'].unique().tolist()
facility_type = df['facility_type'].unique().tolist()
energy_star_rating = df['energy_star_rating'].unique().tolist()
year_built = sorted(df['year_built'].unique().tolist())

