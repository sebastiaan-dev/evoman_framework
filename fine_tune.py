import sys, os

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import pandas as pd
import matplotlib.pyplot as plt
import deap_eaMuPlusLambda as EA2
import deap_eaSimple as E1
import random
def collect_results(name, best_fitness, mean_fitness, sd, gen_to_converge, results):
    results.append({
        'Configuration': name,
        'Best Fitness': best_fitness,
        'Mean Fitness': mean_fitness,
        'Standard Deviation': sd,
        'Generations to Converge': gen_to_converge
    })

def plot_results(df_results):
    # Plot fitness over generations
    df_results.plot(x='Configuration', y=['Best Fitness', 'Mean Fitness'], kind='bar')
    plt.title('Fitness Comparison of Different Configurations')
    plt.ylabel('Fitness')
    plt.xlabel('Configuration')
    plt.legend(['Best Fitness', 'Mean Fitness'])
    plt.show()

    # Plot generations to converge
    df_results.plot(x='Configuration', y='Generations to Converge', kind='bar', color='orange')
    plt.title('Generations to Converge')
    plt.ylabel('Generations')
    plt.xlabel('Configuration')
    plt.show()
alg = 2
def rand_search(env, num_trials=10):
    results = []
    if alg == 1:
        for i in range(num_trials):
            # Sample hyperparameters
            mu = random.choice([50, 100, 200])
            lambda_ = random.choice([50, 100, 200])
            cxpb = random.uniform(0.8, 0.6)
            mutpb = random.uniform(0.1, 0.2)
            ngen = random.choice([20, 30, 50])
            
            name = f"Hyperparameters_{i+1}"
            print(f"Trial {i+1}/{num_trials}: {name} - mu={mu}, lambda_={lambda_}, cxpb={cxpb}, mutpb={mutpb}, ngen={ngen}")
            
            # Run ea
            EA2.run_EA2(env, mu=mu, lambda_=lambda_, cxpb=cxpb, mutpb=mutpb, ngen=ngen, experiment_name=name)

            with open(f"{name}/results.txt", 'r') as log_file:
                lines = log_file.readlines()
                last_line = lines[-1]
                best_fitness = float(last_line.split()[1])
                mean_fitness = float(last_line.split()[2])
                sd = float(last_line.split()[3])
                gen_to_converge = len(lines) - 1
                collect_results(name, best_fitness, mean_fitness, sd, gen_to_converge, results)
            df_results = pd.DataFrame(results)
    elif alg == 2:
        for i in range(num_trials):
            # Sample hyperparameters
            npop = random.choice([50, 100, 200])
            cxpb = random.uniform(0.8, 0.6)
            mutpb = random.uniform(0.1, 0.2)
            ngen = random.choice([20, 30, 50])
            
            name = f"Hyperparameters_EA{alg}_{i+1}"
            print(f"Trial {i+1}/{num_trials}: {name} - npop={npop}, cxpb={cxpb}, mutpb={mutpb}, ngen={ngen}")
            
            # Run ea
            E1.run_eaSimple(env, npop=npop, cxpb=cxpb, mutpb=mutpb, ngen=ngen, experiment_name=name)

            with open(f"{name}/results.txt", 'r') as log_file:
                lines = log_file.readlines()
                last_line = lines[-1]
                best_fitness = float(last_line.split()[1])
                mean_fitness = float(last_line.split()[2])
                sd = float(last_line.split()[3])
                gen_to_converge = len(lines) - 1
                collect_results(name, best_fitness, mean_fitness, sd, gen_to_converge, results)
            df_results = pd.DataFrame(results)
    print("\nSummary:\n", df_results)
    plot_results(df_results)
n_hidden_neurons = 10

# Set up the environment for enemy 1
env = Environment(experiment_name="deneme", enemies=[1],
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)
rand_search(env, num_trials=10)
