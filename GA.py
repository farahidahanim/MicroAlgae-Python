import random
import numpy as np
from deap import base, creator, tools, algorithms 


import pandas as pd

# Load dataset (modify path as needed)
file_path = "C:/Users/Farahida Hanim/OneDrive/Desktop/Experimental Result Algae/Kaggle Dataset Algeas Grow in Artificial Water Bodies/algeas_normalized.csv"
df = pd.read_csv(file_path)

# Display first few rows
#print(df.head()) 


# # Define target Population value
TARGET_POPULATION = 500  # Change this to your desired value 

# # Define the number of variables (excluding Population)
NUM_VARIABLES = 7  # Adjust based on your dataset 
# Bounds for each variable (modify as needed)
VARIABLE_BOUNDS = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)] * 7  #varaible bounds (0,1) is the max and min that you want to achieve, 7 is the number of variable 
# Define how Population is calculated (replace with your actual equation)
def population_model(vars):
    Light,Nitrate,Iron,Phosphate,Temperature,pH,CO2 = vars  # Adjust based on number of variables
    return Light * 1.5 + Nitrate * 2 + Iron * 0.5 + Phosphate *0.3 + Temperature*0.1+ pH*0.1+CO2*0.4# Example model

# # Fitness function: minimize the absolute error to TARGET_POPULATION
# def fitness_function(individual):
#     predicted_population = population_model(individual)
#     error = abs(predicted_population - TARGET_POPULATION)
#     return (1 / (1 + error),)  # Inverse of error (maximize fitness)

# # GA Setup
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)

# toolbox = base.Toolbox()
# toolbox.register("attr_float", random.uniform, 0, 100)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_VARIABLES)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# # Genetic Algorithm Operators
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("evaluate", fitness_function)

# # Run GA
# def run_ga():
#     pop = toolbox.population(n=50)  # Population size
#     algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=True)

#     # Get best individual
#     best_ind = tools.selBest(pop, 1)[0]
#     print(f"Best Input Values: {best_ind}")
#     print(f"Predicted Population: {population_model(best_ind)}")

# run_ga()