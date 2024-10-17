import os
from a2.deap_eaMuPlusLambda import run_muPlusLambda_generalist
from a2.deap_eaSimple import run_eaSimple_generalist
from a2.deap_nsga3 import run_nsga3_generalist
from demo_controller import player_controller
from evoman.environment import Environment
from a2.utils import create_directory_structure

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def simulate_ea_nsga3(args):
    """
    Setup the simulation environment and run the NSGA3 algorithm.
    """
    (
        report,
        experiment_name,
        enemy_group,
        n_hidden_neurons,
        npop,
        ngen,
        cxpb,
        mutpb,
    ) = args

    enemy_dir = create_directory_structure(experiment_name, enemy_group)

    env = Environment(
        randomini="yes",
        multiplemode="yes",
        experiment_name=enemy_dir,
        enemies=enemy_group,
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        logs="off",
    )

    env.state_to_log()

    return run_nsga3_generalist(
        env,
        report,
        npop,
        ngen,
        cxpb,
        mutpb,
        experiment_name,
    )


def simulate_ea_simple(args):
    """
    Setup the simulation environment and run the EA Simple algorithm.
    """
    (
        report,
        experiment_name,
        enemy_group,
        n_hidden_neurons,
        npop,
        ngen,
        cxpb,
        mutpb,
    ) = args

    enemy_dir = create_directory_structure(experiment_name, enemy_group)

    env = Environment(
        randomini="yes",
        multiplemode="yes",
        experiment_name=enemy_dir,
        enemies=enemy_group,
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        logs="off",
    )

    env.state_to_log()

    return run_eaSimple_generalist(
        env,
        report,
        npop,
        ngen,
        cxpb,
        mutpb,
        experiment_name,
    )


def simulate_ea_mupluslambda(args):
    """
    Setup the simulation environment and run the EA Mu Plus Lambda algorithm.
    """
    (
        report,
        experiment_name,
        enemy_group,
        n_hidden_neurons,
        npop,
        ngen,
        lambda_,
        cxpb,
        mutpb,
    ) = args

    enemy_dir = create_directory_structure(experiment_name, enemy_group)

    env = Environment(
        randomini="yes",
        multiplemode="yes",
        experiment_name=enemy_dir,
        enemies=enemy_group,
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="normal",
        visuals=True,
        logs="off",
    )

    env.state_to_log()

    return run_muPlusLambda_generalist(
        env,
        report,
        npop,
        ngen,
        lambda_,
        cxpb,
        mutpb,
        experiment_name,
    )


# Call this file to test a simulation manually.
if __name__ == "__main__":
    simulate_ea_nsga3(
        (
            lambda x: None,
            "test34",
            [1, 4],
            10,
            200,
            40,
            0.8,
            0.15,
        )
    )
