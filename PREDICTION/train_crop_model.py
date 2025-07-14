import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv('datafile.csv')
data = data.dropna()

# Encode the Crop column
data['Crop_encoded'] = data['Crop'].astype('category').cat.codes

# Use 2004-05 to 2010-11 plus Crop_encoded to predict 2011-12
X = data[['2004-05', '2005-06', '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', 'Crop_encoded']]
y = data['2011-12']

# Train the model on ALL the data (no split)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'crop_production_model.pkl')
print("✅ Model trained and saved as 'crop_production_model.pkl'")
