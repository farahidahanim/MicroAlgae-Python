#objective 1 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load data
#file_path = "C:/Users/Farahida Hanim/OneDrive - Universiti Teknologi PETRONAS/algeas.csv"  

#normalized_data 
file_path = "C:/Users/Farahida Hanim/OneDrive/Desktop/Experimental Result Algae/Kaggle Dataset Algeas Grow in Artificial Water Bodies/algeas_normalized.csv"
#get the data
df = pd.read_csv(file_path) 


# Check for missing values
#df.dropna(inplace=True) 
# Define features and target 

X = df[['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2']]
y = df['Population'] 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#random state can be put any value  

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train) 

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'R2 Score: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')  


# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


# Plot feature importance
# plt.figure(figsize=(10,5))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Impact of Features on Population')
# plt.show()

# # Display full feature importance ranking
print("\nFeature Importance Ranking:")
print(feature_importance_df)

# Show predictions vs actual values
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nSample Predictions:")
print(result_df.head(10))  # Display first 10 predictions


# ##Plot actual vs predicted values
# plt.figure(figsize=(10,5))
# plt.scatter(range(len(y_test)), y_test, alpha=0.5, color='red', label='Actual')
# plt.scatter(range(len(y_pred)), y_pred, alpha=0.5, color='blue', label='Predicted')
# plt.xlabel('Sample Index')
# plt.ylabel('Population')
# plt.title('Actual vs Predicted Population')
# plt.legend()
# plt.show()

# Plot actual vs predicted values against Light, temperature and pH and CO2 etc
# plt.figure(figsize=(10,5))
# plt.scatter(X_test['Phosphate'], y_test, alpha=0.5, color='red', label='Actual')
# plt.scatter(X_test['Phosphate'], y_pred, alpha=0.5, color='blue', label='Predicted')
# plt.xlabel('Phosphate')
# plt.ylabel('Population')
# plt.title('Actual vs Predicted Population vs Phosphate')
# plt.legend()
# plt.show()
