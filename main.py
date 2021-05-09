import numpy as np
from settings import *
import traffic_jam_functions as tjf
import matplotlib.pyplot as plt

# initialize the CA
cellular_automaton = np.ones((1, n_rows, n_cols)) * (-1)
cellular_automaton[0, :, 0] = np.random.randint(0, max_model_speed, n_rows)

# evolve the cellular automaton for n_time_steps time steps
cellular_automaton = tjf.evolve2d(cellular_automaton, timesteps=n_time_steps,
                                  apply_rule=lambda n, c, t: tjf.traffic_jam_rule(n, c, t), r=radius)

tjf.plot2d_animate(cellular_automaton)
# for i in range(1, 5):
#    tjf.plot2d(cellular_automaton, i)

plt.show()
