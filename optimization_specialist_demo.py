# imports framework
import sys
import os
import time
import numpy as np
from deap import tools
from deap_eaSimple import run_eaSimple
from evoman.environment import Environment
from demo_controller import player_controller


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def create_directory_structure(experiment_name, enemy_num, run_num):
    """
    Creates a directory structure for saving results based on the enemy and run number.
    Example: optimization_test_eaSimple/E1/run1/ etc.
    """
    enemy_dir = f"{experiment_name}/E{enemy_num}/run{run_num}"
    if not os.path.exists(enemy_dir):
        os.makedirs(enemy_dir)
    return enemy_dir

# Experiment parameters
experiment_name = 'optimization_test_eaSimple'
n_hidden_neurons = 10
n_runs = 10  
enemies = [1, 2, 3]  

# Set the EA function to run (CHANGE TO RUN eaMuPlusLambda)
ea_function = run_eaSimple 

# Initialize simulation in individual evolution mode, for each enemy separately.
for enemy in enemies:
    for run_num in range(1, n_runs + 1):
        print(f"\nRUNNING EVOLUTION WITH eaSimple FOR ENEMY {enemy}, RUN {run_num}\n")

        # create directory for saving the results (e.g. E1/run1)
        enemy_dir = create_directory_structure(experiment_name, enemy, run_num)

        # Initialize environment, save results in enemy-specific dir
        env = Environment(experiment_name=enemy_dir,  
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)

        # default environment fitness is assumed for the experiment
        env.state_to_log()  # Checks environment state

        # If in test mode, load and test the best solution
        run_mode = 'train'  # Set to 'test' if you want to run the best solution [NOT COMPLETE!!!]
        
        if run_mode == 'test':
            best_solution_path = f"{enemy_dir}/best.txt"
            if not os.path.exists(best_solution_path):
                print(f"Error: {best_solution_path} not found.")
                sys.exit(1)

            best_solution = np.loadtxt(best_solution_path)
            print('\nRUNNING SAVED BEST SOLUTION\n')
            env.update_parameter('speed', 'normal')
            env.play(pcont=best_solution)
            sys.exit(0)

        # In train mode: run the EA using your eaSimple function
        else:
            start_time = time.time() 

            ea_function(env, npop=100, ngen=30, cxpb=0.6, mutpb=0.2, experiment_name=enemy_dir)

            end_time = time.time()
            execution_time = round((end_time - start_time) / 60, 2)
            print(f"Run {run_num} for enemy {enemy} completed in {execution_time} minutes.\n")

