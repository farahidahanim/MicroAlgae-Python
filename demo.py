import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV

# Load the data
file_path = "C:/Users/Farahida Hanim/OneDrive - Universiti Teknologi PETRONAS/algeas.csv"
df = pd.read_csv(file_path) 

# Create numpy arrays for features and target
X = df.drop('Population', axis=1).values
y = df['Population'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test) 

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': np.arange(50, 201, 10),  # Number of trees in the forest
    'max_depth': np.arange(10, 101, 10),    # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],       # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],         # Minimum samples required to be a leaf node
    'bootstrap': [True, False]             # Whether bootstrap samples are used
}
# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor() 

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_grid, n_iter=100,
                                   scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1)
# Fit the RandomizedSearchCV to your data
random_search.fit(X_scaled, y_train) 

# Print the best hyperparameters found
print("Best Hyperparameters:", random_search.best_params_) 

# Get the best model from the search
best_rf_model = random_search.best_estimator_

# Evaluate the best model on your test set
y_pred = best_rf_model.predict(X_scaled_test)

# Calculate evaluation metrics (e.g., MSE, MAE, R2) on the test set
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r_squared:.2f}")