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
            if not line or line.startswith('gen'):  
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

def find_best_run(results_dir):
    """ Find the run with the best final fitness in the last generation """
    best_run = None
    best_fitness = -np.inf

    for run in range(1, 11):  # Assuming 10 runs
        result_path = os.path.join(results_dir, f'run{run}', 'results.txt')
        if not os.path.exists(result_path):
            continue

        generations, best_fitnesses, average_fitnesses, std_devs = read_results(result_path)

        # Check the last generation's max fitness
        final_best_fitness = best_fitnesses[-1]
        if final_best_fitness > best_fitness:
            best_fitness = final_best_fitness
            best_run = run

    return best_run

def visualize_best_runs(ea_simple_dir, ea_mu_dir, enemy):
    """ Visualizes the best runs (max fitness individual) of each EA for a given enemy """
    # Find the best run for eaSimple
    best_simple_run = find_best_run(os.path.join(ea_simple_dir, f'E{enemy}'))
    best_mu_run = find_best_run(os.path.join(ea_mu_dir, f'E{enemy}'))

    if best_simple_run is None or best_mu_run is None:
        print(f"Could not find results for one of the EAs for Enemy {enemy}.")
        return

    # Results of the best run for eaSimple
    simple_result_path = os.path.join(ea_simple_dir, f'E{enemy}', f'run{best_simple_run}', 'results.txt')
    generations_simple, best_fitnesses_simple, average_fitnesses_simple, std_devs_simple = read_results(simple_result_path)

    # Results of the best run for eaMuPlusLambda
    mu_result_path = os.path.join(ea_mu_dir, f'E{enemy}', f'run{best_mu_run}', 'results.txt')
    generations_mu, best_fitnesses_mu, average_fitnesses_mu, std_devs_mu = read_results(mu_result_path)

    # Plot the best runs for both EAs
    plt.figure(figsize=(12, 6))

    plt.plot(generations_simple, best_fitnesses_simple, label='Best Fitness (eaSimple)', color='blue', marker='o')
    plt.plot(generations_simple, average_fitnesses_simple, label='Average Fitness (eaSimple)', color='orange', marker='x')
    plt.fill_between(generations_simple, 
                     np.array(average_fitnesses_simple) - np.array(std_devs_simple), 
                     np.array(average_fitnesses_simple) + np.array(std_devs_simple), 
                     color='green', alpha=0.3, label='Std Dev (eaSimple)')

    plt.plot(generations_mu, best_fitnesses_mu, label='Best Fitness (eaMuPlusLambda)', color='red', marker='o')
    plt.plot(generations_mu, average_fitnesses_mu, label='Average Fitness (eaMuPlusLambda)', color='purple', marker='x')
    plt.fill_between(generations_mu, 
                     np.array(average_fitnesses_mu) - np.array(std_devs_mu), 
                     np.array(average_fitnesses_mu) + np.array(std_devs_mu), 
                     color='yellow', alpha=0.3, label='Std Dev (eaMuPlusLambda)')

    plt.title(f'Comparison of Best Run for eaSimple and eaMuPlusLambda (Enemy {enemy})')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc='lower right')
    plt.grid()

    save_dir = 'visualizations/best_runs_comparison_per_enemy'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(os.path.join(save_dir, f'best_run_comparison_enemy_{enemy}.png'))
    plt.show()

if __name__ == "__main__":
    ea_simple_dir = 'runs_eaSimple'
    ea_muPlusLambda_dir = 'runs_muPlusLambda'
    enemies = [1, 2, 3]  #  Enemies to visualize

    for enemy in enemies:
        visualize_best_runs(ea_simple_dir, ea_muPlusLambda_dir, enemy)
