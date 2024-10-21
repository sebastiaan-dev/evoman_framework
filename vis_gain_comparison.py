import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evoman.environment import Environment
from demo_controller import player_controller

# Settings for evoman
os.environ["SDL_VIDEODRIVER"] = "dummy"

repetitions = 5
speed = "fastest"
fullscreen = False
sound = "off"
visuals = False

n_enemies = 3
n_hidden = 10

enemy_groups = [[1, 2, 3]]  # [4, 7, 8]]


# Calculate the mean gain for a specific enemy and run
def calc_mean_gain(best_solution, enemy):
    print(f"Calculating mean gain for enemy {enemy}")
    # Initialize the environment
    env = Environment(
        experiment_name="test",
        enemies=[enemy],
        playermode="ai",
        fullscreen=fullscreen,
        player_controller=player_controller(n_hidden),
        enemymode="static",
        level=2,
        sound=sound,
        speed=speed,
        visuals=visuals,
    )

    # Calculate the gain for each repetition
    gains = []
    for _ in range(repetitions):
        f, p, e, t = env.play(pcont=best_solution)
        gains.append(p - e)

    return gains


# Calculate the mean gain for each enemy and run
def calc_mean_gains(best_solution, enemy_group):
    results = {}

    for enemy in enemy_group:
        results[enemy] = calc_mean_gain(best_solution, enemy)

    return results


# Calculate the mean gains for both algorithms
def calc_mean_gains_for_algorithms(enemy_group):
    # Load the best solutions for both algorithms
    enemy_group_dir = "_".join(map(str, enemy_group))

    nsga3_best_solution = np.loadtxt(f"best/nsga3/{enemy_group_dir}/best.txt")
    eaLambda_best_solution = np.loadtxt(f"best/mupluslambda/{enemy_group_dir}/best.txt")

    nsga3_gains = calc_mean_gains(nsga3_best_solution, enemy_group)
    eaLambda_gains = calc_mean_gains(eaLambda_best_solution, enemy_group)

    return nsga3_gains, eaLambda_gains


for enemy_group in enemy_groups:
    aSimple_gains, eaLambda_gains = calc_mean_gains_for_algorithms(enemy_group)

    # Make a boxplot which compares the gains for both algorithms per enemy
    fig, ax = plt.subplots()
    ax.boxplot(
        aSimple_gains.values(), positions=np.arange(len(enemy_group)) - 0.2, widths=0.4
    )
    ax.boxplot(
        eaLambda_gains.values(), positions=np.arange(len(enemy_group)) + 0.2, widths=0.4
    )
    ax.set_xticks(range(len(enemy_group)))
    ax.set_xticklabels(enemy_group)
    ax.set_xlabel("Enemy")
    ax.set_ylabel("Mean Gain")
    ax.set_title(f"Mean Gain for Enemies {enemy_group}")
    ax.legend(["Simple", "MuPlusLambda"])
    # Set the y-axis to start at -100 and end at 100
    ax.set_ylim(-100, 100)

    plt.show()
#     plt.savefig(f"visualizations/gains_comparison_{enemy_group}.png")
# nsga3_enemy_gains = [aSimple_gains[enemy] for enemy in enemy_group]
# eaLambda_enemy_gains = [eaLambda_gains[enemy] for enemy in enemy_group]

# print(nsga3_enemy_gains)

# fig, ax = plt.subplots()
# ax.boxplot(
#     [nsga3_enemy_gains, eaLambda_enemy_gains],
#     labels=["nsga3", "MuPlusLambda"],
# )

# enemy_group = " ".join(map(str, enemy_group))
# ax.set_title(f"Enemies {enemy_group}")
# ax.set_ylabel("Mean gain")
# enemy_group = "".join(map(str, enemy_group))
# plt.savefig(f"boxplot_gain_enemy_{enemy_group}.png", dpi=300)
