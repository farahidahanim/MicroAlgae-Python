import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#get data

# Create DataFrame
file_path = "C:/Users/Farahida Hanim/OneDrive - Universiti Teknologi PETRONAS/algeas.csv"
df = pd.read_csv(file_path)  


# Basic Summary Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Correlation Matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Data Visualization
plt.figure(figsize=(12, 5))


# # Histogram of Temperature
# plt.subplot(1, 3, 1)
# sns.histplot(df['Light'], bins=5, kde=True, color='blue')
# plt.title('Light Distribution')

#Boxplot for Growth Rate
plt.subplot(1, 3, 2)
sns.boxplot(y=df['Population'], color='green')
plt.title('Boxplot Rate')

# # Scatter Plot for Temperature vs Growth Rate
# plt.subplot(1, 3, 3)
# sns.scatterplot(x=df['Temperature'], y=df['Growth_Rate'], hue=df['pH'], palette='coolwarm')
# plt.title('Temperature vs Growth Rate')

plt.tight_layout()
plt.show()
