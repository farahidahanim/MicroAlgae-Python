
#multiple linear regression 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Sample Data (Replace with your actual dataset)

file_path = "C:/Users/Farahida Hanim/OneDrive - Universiti Teknologi PETRONAS/algeas.csv"
df = pd.read_csv(file_path) 

# Initialize Min-Max Scaler--------------------------------------
scaler = MinMaxScaler() 
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns) 
# Independent variables (X) - Features
X = df_normalized[['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2']]

# Dependent variable (y) - Target
y = df_normalized['Population'] 

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test) 

# Calculate R² score (coefficient of determination)
r2 = r2_score(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Print evaluation metrics
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}") 

# Get feature importance (coefficients)
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients) 


#Visualization -------------------------
 #Feature Importance (Bar Chart)
# plt.figure(figsize=(8, 5))
# sns.barplot(x=coefficients['Feature'], y=coefficients['Coefficient'], palette='viridis')
# plt.title("Feature Importance in Regression Model")
# plt.xticks(rotation=45)
# plt.show()

#Actual vs Predicted Population (Scatter Plot) 
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Population")
plt.ylabel("Predicted Population")
plt.title("Actual vs Predicted Population")
plt.axline([0, 0], slope=1, color="red", linestyle="dashed")  # Perfect fit line
plt.show()