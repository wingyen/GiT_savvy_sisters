import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load your dataset
file_path = './data_set.csv'
data = pd.read_csv(file_path)

# Assuming the columns are named X1, X2, and y
X = data[['building_class', 'facility_type', 'energy_star_rating', 'year_built']]
y = data['site_eui']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model to a file
model_filename = 'demo_model.pkl'
joblib.dump(model, model_filename)
