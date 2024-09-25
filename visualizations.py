import matplotlib.pyplot as plt
import numpy as np
import os

def read_results(file_path):
    generations = []
    best_fitnesses = []
    average_fitnesses = []
    std_devs = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  
            if not line or line.startswith('gen'):  # Skip empty lines and header
                continue
            parts = line.split()
            if len(parts) < 4: 
                continue 

            try:
                gen = int(parts[0])
                best_fit = float(parts[1])
                avg_fit = float(parts[2])
                std_fit = float(parts[3])

                generations.append(gen)
                best_fitnesses.append(best_fit)
                average_fitnesses.append(avg_fit)
                std_devs.append(std_fit)
            except ValueError:
                continue  

    return generations, best_fitnesses, average_fitnesses, std_devs

# Visualization function for two EAs
def visualize_comparison(generations_simple, best_fitnesses_simple, average_fitnesses_simple, std_devs_simple,
                         generations_mu, best_fitnesses_mu, average_fitnesses_mu, std_devs_mu):
    
    plt.figure(figsize=(12, 6))

    # Plot for eaSimple
    plt.plot(generations_simple, best_fitnesses_simple, label='Best Fitness (eaSimple)', color='blue', marker='o')
    plt.plot(generations_simple, average_fitnesses_simple, label='Average Fitness (eaSimple)', color='orange', marker='x')
    plt.fill_between(generations_simple, 
                     np.array(average_fitnesses_simple) - np.array(std_devs_simple), 
                     np.array(average_fitnesses_simple) + np.array(std_devs_simple), 
                     color='green', alpha=0.3, label='Std Dev (eaSimple)')
    
    # Plot for eaMuPlusLambda
    plt.plot(generations_mu, best_fitnesses_mu, label='Best Fitness (eaMuPlusLambda)', color='red', marker='o')
    plt.plot(generations_mu, average_fitnesses_mu, label='Average Fitness (eaMuPlusLambda)', color='purple', marker='x')
    plt.fill_between(generations_mu, 
                     np.array(average_fitnesses_mu) - np.array(std_devs_mu), 
                     np.array(average_fitnesses_mu) + np.array(std_devs_mu), 
                     color='yellow', alpha=0.3, label='Std Dev (eaMuPlusLambda)')
    

    plt.title('Comparison of Fitness Over Generations for eaSimple and eaMuPlusLambda')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":

    result_file_path_simple = 'dummy_demo_eaSimple/results.txt'
    result_file_path_mu = 'dummy_demo_muPlusLambda/results.txt'

    generations_simple, best_fitnesses_simple, average_fitnesses_simple, std_devs_simple = read_results(result_file_path_simple)
    generations_mu, best_fitnesses_mu, average_fitnesses_mu, std_devs_mu = read_results(result_file_path_mu)
    
    # Visualize comparison
    visualize_comparison(generations_simple, best_fitnesses_simple, average_fitnesses_simple, std_devs_simple,
                         generations_mu, best_fitnesses_mu, average_fitnesses_mu, std_devs_mu)
