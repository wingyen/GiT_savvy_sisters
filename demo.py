import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from helpers import building_class, facility_type, energy_star_rating, year_built


st.title('Savvy Sisters Green Buildings')


# Load the trained model
model = xgb.XGBRegressor()
model.load_model('./best_model_41_6.json')
# model_path = './demo_model.pkl'
# model = joblib.load(model_path)  # Load the trained model

# Dropdown options for building_class, facility_type, and energy_star_rating
building_class_options = building_class  # Replace with your options
facility_type_options = facility_type  # Replace with your options
#energy_star_rating_options = energy_star_rating  # Replace with your options

# Input fields for user input
building_class = st.selectbox('Select Building Class:', building_class_options)
facility_type = st.selectbox('Select Facility Type:', facility_type_options)
#energy_star_rating = st.selectbox('Select Energy Star Rating:', energy_star_rating_options)
year_built = st.number_input('Enter Year Built:', min_value=int(min(year_built)), max_value=int(max(year_built)), value=2015)



def preprocess_dataframe(input_df):
    df = input_df.copy()
    # Handling missing values
    df['year_built'] = df['year_built'].replace(np.nan, 2022)
    null_col = ['energy_star_rating']
    df[null_col] = SimpleImputer.fit_transform(df[null_col])
    
    # Label encoding for categorical features
    categorical_features = ['State_Factor', 'building_class', 'facility_type']
    le = LabelEncoder()
    for col in categorical_features:
        df[f'{col}_label'] = le.fit_transform(df[col])
    
    # One-hot encoding for 'building_class'
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(df[['building_class']])
    one_hot_encoded = enc.transform(df[['building_class']])
    encoded_df = pd.DataFrame(one_hot_encoded, columns=['is_commercial', 'is_residential'])
    df['is_commercial'] = encoded_df['is_commercial']
    df['is_residential'] = encoded_df['is_residential']
    
    # Extracting and encoding facility types
    df['facility_type_2'] = df['facility_type'].apply(lambda x: '_'.join(x.split('_')[1:]))
    df['facility_type_1'] = df['facility_type'].apply(lambda x: x.split('_')[0])
    facility_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    facility_enc.fit(df[['facility_type_1']])
    one_hot_facility_type_1 = facility_enc.transform(df[['facility_type_1']])
    for i, category in enumerate(facility_enc.categories_[0]):
        df[f'is_{category}'] = one_hot_facility_type_1[:, i]
    facility_enc.fit(df[['facility_type_2']])
    one_hot_facility_type_2 = facility_enc.transform(df[['facility_type_2']])
    for i, category in enumerate(facility_enc.categories_[0]):
        df[f'is_{category}'] = one_hot_facility_type_2[:, i]
    # Group by and add mean values
    mean_values_1 = df.groupby('facility_type_1')['site_eui'].transform('mean')
    df['mean_value_by_facility_type_1'] = mean_values_1
    mean_values_2 = df.groupby('facility_type_2')['site_eui'].transform('mean')
    df['mean_value_by_facility_type_2'] = mean_values_2
    return df

# When 'Predict' button is clicked
if st.button('Predict'):
    # Preprocess user input
    input_data = pd.DataFrame({
        'building_class': [building_class],
        'facility_type': [facility_type],
        #'energy_star_rating': [energy_star_rating],
        'year_built': [float(year_built)]
    })

    # Perform one-hot encoding for categorical variables
    input_data_encoded = pd.get_dummies(input_data)  # Convert categorical variables into numerical format
    
    # Ensure input columns match model input
    model_feature_names = model.feature_names if hasattr(model, 'feature_names') else list(input_data_encoded.columns)
    #input_data_processed = input_data_encoded.reindex(columns=model_feature_names, fill_value=0)
    input_data_processed = preprocess_dataframe(input_data_encoded)
    # Convert input data to DMatrix format
    input_dmatrix = xgb.DMatrix(data=input_data_processed.values)
    
    # Make prediction using the loaded model
    prediction = model.predict(input_dmatrix)
    
    
    # Display prediction result
    st.write(f"Predicted Output: {prediction[0]}")
    st.write("Note: The ")