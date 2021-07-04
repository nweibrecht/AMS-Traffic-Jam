import numpy as np
from settings import *
import traffic_jam_functions as tjf
import matplotlib.pyplot as plt

# initialize the CA
np.random.seed(seed)
cellular_automaton = np.ones((n_rows, n_cols, 2)) * (-1)
cellular_automaton[:, 0, 0] = np.random.randint(0, max_model_speed, n_rows)
cellular_automaton[:, 0, 1] = [i for i in range(n_rows)]
cellular_automaton = np.ones((n_rows, n_cols, 2)) * (-1)

for row in range(n_rows):
    if tjf.is_blocked(row, 0):
        cellular_automaton[row, 0, 0] = np.random.randint(0, max_model_speed)
        cellular_automaton[row, 0, 1] = row

# evolve the cellular automaton for n_time_steps time steps
cellular_automaton = tjf.evolve2d(cellular_automaton, timesteps=n_time_steps,
                                  apply_rule=lambda n, c, t: tjf.traffic_jam_rule(n, c, t), r=radius)
values = cellular_automaton[:, :, :, 0]
car_numbers = cellular_automaton[:, :, :, 1]
tjf.saveImages(values, car_numbers)
plt.show()

best_pams, min_difference = tjf.parametrize_goal_time(9.5, max_model_speed)
print(best_pams, min_difference)
