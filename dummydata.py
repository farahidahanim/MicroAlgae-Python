import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Function to generate random data for a dataset
def generate_microalgae_dataset(num_rows, num_columns):
    # Generate random dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_rows)]
    
    # Generate random values for variables
    temperature = np.random.uniform(15, 35, num_rows)  # Temperature between 15 and 35°C
    light_intensity = np.random.uniform(200, 2000, num_rows)  # Light intensity between 200 and 2000 lux
    pH_level = np.random.uniform(6.0, 9.0, num_rows)  # pH level between 6 and 9
    nitrate_concentration = np.random.uniform(0, 10, num_rows)  # Nitrate concentration in mg/L
    phosphate_concentration = np.random.uniform(0, 5, num_rows)  # Phosphate concentration in mg/L
    ammonium_concentration = np.random.uniform(0, 5, num_rows)  # Ammonium concentration in mg/L
    salinity = np.random.uniform(0, 35, num_rows)  # Salinity between 0 and 35 ppt
    growth_rate = np.random.uniform(0.5, 3.0, num_rows)  # Growth rate between 0.5 and 3 cells/mL/day
    chlorophyll_a_concentration = np.random.uniform(0, 50, num_rows)  # Chlorophyll-a concentration
    turbidity = np.random.uniform(0, 100, num_rows)  # Turbidity in NTU
    dissolved_oxygen = np.random.uniform(4, 10, num_rows)  # Dissolved oxygen in mg/L
    co2_concentration = np.random.uniform(200, 500, num_rows)  # CO2 concentration in ppm
    humidity = np.random.uniform(40, 90, num_rows)  # Humidity in percentage
    seasonal_variation = [random.choice(['Spring', 'Summer', 'Fall', 'Winter']) for _ in range(num_rows)]
    
    # Combine data into a DataFrame
    data = {
        "Date": dates,
        "Temperature (°C)": temperature,
        "Light Intensity (lux)": light_intensity,
        "pH Level": pH_level,
        "Nitrate Concentration (mg/L)": nitrate_concentration,
        "Phosphate Concentration (mg/L)": phosphate_concentration,
        "Ammonium Concentration (mg/L)": ammonium_concentration,
        "Salinity (ppt)": salinity,
        "Growth Rate (cells/mL/day)": growth_rate,
        "Chlorophyll-a concentration (μg/L)": chlorophyll_a_concentration,
        "Water turbidity (NTU)": turbidity,
        "Dissolved Oxygen (mg/L)": dissolved_oxygen,
        "CO2 Concentration (ppm)": co2_concentration,
        "Humidity (%)": humidity,
        "Seasonal Variation": seasonal_variation
    }

    # Adding more columns
    for i in range(num_columns - 15):  # 15 columns already created
        col_name = f"Variable_{i + 16}"
        data[col_name] = np.random.uniform(0, 10, num_rows)

    df = pd.DataFrame(data)
    return df

# Create two datasets
dataset1 = generate_microalgae_dataset(10000, 300)
dataset2 = generate_microalgae_dataset(10000, 300)

# Save the datasets as CSV files
dataset1.to_csv("C:/Users/Farahida Hanim/MicroAlgae Python/microalgae_growth_data.csv", index=False)
dataset2.to_csv("C:/Users/Farahida Hanim\MicroAlgae Python/microalgae_production_data.csv", index=False)

print("Datasets created and saved as microalgae_growth_data.csv and microalgae_production_data.csv")
