from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np

repetitions = 1
speed = "normal"
fullscreen = False
sound = "off"
visuals = True

n_enemies = 3
n_hidden = 10


env = Environment(
    experiment_name="test",
    enemies=[1],
    playermode="ai",
    fullscreen=fullscreen,
    player_controller=player_controller(n_hidden),
    enemymode="static",
    level=2,
    sound=sound,
    speed=speed,
    visuals=visuals,
)

# path = "runs_eaSimple/E1/run1/best.txt"
path = "runs_muPlusLambda/E1/run1/best.txt"

f, p, e, t = env.play(pcont=np.loadtxt(path))

print(f"Gain for player: {p - e}")
