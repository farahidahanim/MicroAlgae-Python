#Objective : To Determining Key Influencing Factors in Regression Anysis : : Identify which variables (e.g., nutrients, pH, temperature) most significantly impact population growth.
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = "C:/Users/Farahida Hanim/OneDrive - Universiti Teknologi PETRONAS/algeas.csv"
df = pd.read_csv(file_path) 

# Initialize Min-Max Scaler--------------------------------------
scaler = MinMaxScaler() 
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  

# save normalized data to file----------------------------------------
# # Save to CSV (default: includes index)
# df_normalized.to_csv("C:/Users/Farahida Hanim/OneDrive/Desktop/Experimental Result Algae/Kaggle Dataset Algeas Grow in Artificial Water Bodies/algeas_normalized.csv", index=False)
# print("CSV file saved successfully!") 

#Exploratory Data Analysis 
# import seaborn as sns
# import matplotlib.pyplot as plt 
# # Correlation matrix 
# correlation_matrix = df_normalized.corr() 
# # Plot heatmap
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.show() 

#Step 3. Run Multiple Linear Regression------------------
import statsmodels.api as sm

#Define independent variables (X) and dependent variable (y)
X = df_normalized[['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2']]
y = df_normalized['Population']

# Add constant for intercept
#X = sm.add_constant(X)

# Run regression
model = sm.OLS(y, X).fit()

# Print results
#print(model.summary()) #----------run this if want to see the reults of the regression analysis

# --------------------Feature Selection ------------------
#want to remove less significant variable
#ada beberapa cara untuk buat feature selection : 
#1.Stepwise Regression
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# rfe = RFE(model, n_features_to_select=3)  # Select top 3 features
# fit = rfe.fit(X, y)
# print(fit.support_)  # Shows selected features
# print(fit.ranking_)  # Ranking of feature importance 

#3. Random Forest Feature Importance 
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))  