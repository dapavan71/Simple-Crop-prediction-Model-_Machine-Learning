import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('datafile.csv')
print("Original data:")
print(data.head())

# Drop rows with any missing data
data = data.dropna()

# Encode 'Crop'
data['Crop_encoded'] = data['Crop'].astype('category').cat.codes

# Use years 2004-05 to 2010-11 plus Crop_encoded to predict 2011-12
X = data[['2004-05', '2005-06', '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', 'Crop_encoded']]
y = data['2011-12']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual 2011-12 Production')
plt.ylabel('Predicted 2011-12 Production')
plt.title('Actual vs Predicted Production')
plt.grid(True)
plt.show()
