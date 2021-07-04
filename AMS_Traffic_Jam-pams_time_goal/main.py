import numpy as np
from settings import *
import traffic_jam_functions as tjf
import matplotlib.pyplot as plt

# initialize the CA
np.random.seed(seed)
cellular_automaton = np.ones((n_rows, n_cols, 2)) * (-1)
cellular_automaton[:, 0, 0] = np.random.randint(0, max_model_speed, n_rows)
cellular_automaton[:, 0, 1] = [i for i in range(n_rows)]
print(cellular_automaton.shape)
# evolve the cellular automaton for n_time_steps time steps
cellular_automaton = tjf.evolve2d(cellular_automaton, timesteps=n_time_steps,
                                  apply_rule=lambda n, c, t: tjf.traffic_jam_rule(n, c, t), r=radius)
values = cellular_automaton[:,:,:,0]
print(values.shape)
tjf.plot2d_animate(values)
# tjf.saveImages(values)
# for i in range(1, 5):
#    tjf.plot2d(cellular_automaton, i)

# tjf.plot2d(cellular_automaton, 5)
plt.show()


best_pams,min_difference=parametrize_goal_time(9.5,max_model_speed)