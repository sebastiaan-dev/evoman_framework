import matplotlib.pyplot as plt
import os


def read_results(file_path):
    generations = []
    best_fitnesses = []
    average_fitnesses = []
    std_devs = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("gen"):
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


# Visualize results for one EA and one enemy
def visualize_runs_for_enemy_group(ax, enemy_group, ea_name, results_dir):
    """Visualizes runs for a specific enemy in a subplot ax."""
    colors = plt.get_cmap("tab20")

    for run_num in range(1, 11):  # Iterate through 10 runs
        group_string = "_".join(map(str, enemy_group))
        result_file = os.path.join(
            results_dir, f"{group_string}", f"run{run_num}", "result.txt"
        )
        if not os.path.exists(result_file):
            print(f"Results file not found: {result_file}")
            continue

        generations, best_fitnesses, average_fitnesses, std_devs = read_results(
            result_file
        )

        color = colors((run_num - 1) / 20)

        # Plot max fitness: solid line
        ax.plot(
            generations,
            best_fitnesses,
            color=color,
            linestyle="-",
            alpha=0.7,
            label=f"Run {run_num} Max Fitness",
        )

        # Plot mean fitness: dashed line
        ax.plot(
            generations,
            average_fitnesses,
            color=color,
            linestyle="--",
            alpha=0.7,
            label=f"Run {run_num} Mean Fitness",
        )

    group_string = " ".join(map(str, enemy_group))
    ax.set_title(f"Enemies {group_string}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid()


def visualize_ea_for_enemies(ea_name, results_dir, enemy_groups):
    """Generates side-by-side plots of all enemies for one EA."""
    save_dir = "visualizations"
    fig, axs = plt.subplots(1, len(enemy_groups), figsize=(18, 6))

    for i, enemy in enumerate(enemy_groups):
        visualize_runs_for_enemy_group(axs[i], enemy, ea_name, results_dir)

    fig.suptitle(f"Fitness Over Generations for {ea_name} Across Enemy groups")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # combined plot for the EA
    plt.savefig(os.path.join(save_dir, f"{ea_name}_comparison_across_enemies.png"))
    plt.show()


if __name__ == "__main__":
    mupluslambda_dir = "results/mupluslambda"
    nsga3_dir = "results/nsga3"

    # Enemies to visualize
    enemy_groups = [[1, 2, 3], [4, 7, 8]]

    visualize_ea_for_enemies("MuPlusLambda", mupluslambda_dir, enemy_groups)
    visualize_ea_for_enemies("NSGA-III", nsga3_dir, enemy_groups)
