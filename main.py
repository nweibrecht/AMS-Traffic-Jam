import numpy as np
from settings import *
import traffic_jam_functions as tjf
import matplotlib.pyplot as plt

# initialize the CA
np.random.seed(seed)
cellular_automaton = np.ones((n_rows, n_cols, 2)) * (-1)
for row in range(n_rows):
    if tjf.is_blocked(row, 0) :
        cellular_automaton[row, 0, 0] = np.random.randint(0, max_model_speed)
        cellular_automaton[row, 0, 1] = row
print(cellular_automaton.shape)
# evolve the cellular automaton for n_time_steps time steps
cellular_automaton = tjf.evolve2d(cellular_automaton, timesteps=n_time_steps,
                                  apply_rule=lambda n, c, t: tjf.traffic_jam_rule(n, c, t), r=radius)
values = cellular_automaton[:,:,:,0]
print(values.shape)
tjf.saveImages(values)
# tjf.saveImage(values,n_time_steps-1)
plt.show()
