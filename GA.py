
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
file_path = "C:/Users/Farahida Hanim/OneDrive/Desktop/Experimental Result Algae/Kaggle Dataset Algeas Grow in Artificial Water Bodies/algeas_normalized.csv"
#get the data
df = pd.read_csv(file_path) 


X = df['Light','Nitrate','Iron','Phosphate','Temperature','pH','CO2']
y = df['PopulationP'] 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#random state can be put any value  

# Standardize features
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test) 

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) 

# Predictions
y_pred = model.predict(X_test)


# Check model accuracy
y_pred = model.predict(X_test)
print(f"Model R² Score: {r2_score(y_test, y_pred)}")  # Should be close to 1 if well-fitted

#...............................................................................................................
import random
import numpy as np
from deap import base, creator, tools, algorithms  

# Set target value for v4
TARGET_V4 = 0.90211 # Change this to desired value

# Define bounds for variables (Assume values between 0 and 1, adjust based on your data)
VARIABLE_BOUNDS = [(0, 1), (0, 1), (0, 1),(0, 1), (0, 1), (0, 1),(0, 1)]  # v1, v2, v3

# Define fitness function for GA
def fitness_function(individual):
    """Fitness function to minimize the error between predicted and target v4."""
    vars = np.array(individual).reshape(1, -1)  # Convert to correct shape
    predicted_v4 = model.predict(vars)[0]  # Predict using trained model
    error = abs(predicted_v4 - TARGET_V4)
    return (1 / (1 + error),)  # Maximize fitness by minimizing error

# GA Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)  # Variables between (0,1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=7)  # 3 variables
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# GA Operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)

# Run GA
def run_ga():
    pop = toolbox.population(n=50)  # Population size
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=True)

    # Get best solution
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Best Input Values: {best_ind}")
    print(f"Predicted v4: {model.predict([best_ind])[0]}")  # Check final v4 value

run_ga()
