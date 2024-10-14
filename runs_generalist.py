# imports framework
import sys
import os
import time
import numpy as np
import argparse
from deap import tools
from deap_eaMuPlusLambda_generalist import run_muPlusLambda_generalist
from deap_eaSimple_generalist import run_eaSimple_generalist
from evoman.environment import Environment
from demo_controller import player_controller

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def create_directory_structure(experiment_name, enemy_num, run_num):
    """
    Creates a directory structure for saving results based on the enemy and run number.
    Example: optimization_test_eaSimple_generalist/E1/run1/ etc.
    """
    enemy_dir = f"{experiment_name}/E{' '.join(str(e) for e in enemy_num)}/run{run_num}"
    if not os.path.exists(enemy_dir):
        os.makedirs(enemy_dir)
    return enemy_dir

# Experiment parameters
n_hidden_neurons = 10
n_runs = 10  
enemies = [1, 2, 3]  

# Set the EA function to run (CHANGE TO RUN EITHER EA SIMPLE OR MU+LAMBDA)
#ea_function = run_muPlusLambda 
ea_function = sys.argv[1]
enemy_groups = [[1, 2, 3], [4, 5, 6], [1,4]]

function_dict = {
    "simple": run_eaSimple_generalist,
    "muPlusLambda": run_muPlusLambda_generalist
}



# Initialize simulation in individual evolution mode, for each enemy separately.
for enemy_group in enemy_groups:
    for run_num in range(1, n_runs + 1):
        print(f"\nRUNNING EVOLUTION FOR ENEMY {enemy_group}, RUN {run_num}\n")
        
        if ea_function == "simple":
            experiment_name = 'runs_eaSimple_generalist'
        elif ea_function == "muPlusLambda":
            experiment_name = 'run_muPlusLambda_generalist'

        # create directory for saving the results (e.g. E1/run1)
        enemy_dir = create_directory_structure(experiment_name, enemy_group, run_num)

        # Initialize environment, save results in enemy-specific dir
        env = Environment(experiment_name=enemy_dir,  
                          enemies=enemy_group,
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

        # In train mode: run the given EA
        else:
            start_time = time.time() 

            # TODO: Run EAs with optimized parameters
            if ea_function == "simple":
                function_dict["simple"](env, npop=200, ngen=30, cxpb=0.616, mutpb=0.119, experiment_name=enemy_dir)
            elif ea_function == "muPlusLambda":
                function_dict["muPlusLambda"](env, mu=200, ngen=30, lambda_=100, cxpb=0.69, mutpb=0.11, experiment_name=enemy_dir)
    

            end_time = time.time()
            execution_time = round((end_time - start_time) / 60, 2)
           # print(f"Run {run_num} for enemy {enemy} completed in {execution_time} minutes.\n")

