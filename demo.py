import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from helpers import building_class, facility_type, energy_star_rating, year_built


st.title('Savvy Sisters Green Buildings')


# Load the JSON format model
model_path = './best_model_41_6.json.json'
with open(model_path, 'r') as f:
    model_json = json.load(f)

# Load the model from JSON
model = xgb.Booster(model_file=model_json)


# Dropdown options for building_class, facility_type, and energy_star_rating
building_class_options = building_class  # Replace with your options
facility_type_options = facility_type  # Replace with your options
energy_star_rating_options = energy_star_rating  # Replace with your options

# Input fields for user input
building_class = st.selectbox('Select Building Class:', building_class_options)
facility_type = st.selectbox('Select Facility Type:', facility_type_options)
energy_star_rating = st.selectbox('Select Energy Star Rating:', energy_star_rating_options)
year_built = st.number_input('Enter Year Built:', min_value=int(min(year_built)), max_value=int(max(year_built)), value=2015)

# When 'Predict' button is clicked
if st.button('Predict'):
    # Preprocess user input
    input_data = pd.DataFrame({
        'building_class': [building_class],
        'facility_type': [facility_type],
        'energy_star_rating': [energy_star_rating],
        'year_built': [year_built]
    })

    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display prediction result
    st.write(f"Predicted Output: {prediction}")




