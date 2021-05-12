import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from settings import *


def evolve2d(cellular_automaton, timesteps, apply_rule, r=1):
    _, rows, cols = cellular_automaton.shape
    array = np.zeros((timesteps, rows, cols), dtype=cellular_automaton.dtype)
    array[0] = cellular_automaton

    def get_neighbourhood(cell_layer, row, col):
        # Neighborhood of traffic jam:
        col_indices = range(col - r, col + r + 1)
        row_indices = range(row - 1, row + 2)
        row_indices = [i for i in row_indices if 0 <= i < n_rows]
        col_indices = [i for i in col_indices if 0 <= i < n_cols]
        return cell_layer[np.ix_(row_indices, col_indices)]

    for t in range(1, timesteps):
        cell_layer = array[t - 1]
        for row, cell_row in enumerate(cell_layer):
            for col, cell in enumerate(cell_row):
                n = get_neighbourhood(cell_layer, row, col)
                array[t][row][col] = apply_rule(n, (row, col), t)
    return array


def value_is_of_interest(index, cell, next_cell):
    # The value of interest will reach the current cell in the next time step
    car_will_reach_cell = index == cell
    car_has_to_break = next_cell != -1 and cell > index
    return car_will_reach_cell or car_has_to_break


def traffic_jam_rule(neighborhood, c, t):
    (row, col) = c
    if row == 0:
        row_index_of_lane = 0
    else:
        row_index_of_lane = 1
    curr_lane = neighborhood[row_index_of_lane, :]  # lane of the considered cell

    index_of_current_cell = col if col <= radius else radius  # index of current cell within neighborhood
    important_cells = list(curr_lane[:index_of_current_cell + 2])  # cells until one after current cell
    one_after_current_cell = important_cells.pop()  # value after the current cell
    important_cells.reverse()
    result = next(((ind, c) for ind, c in enumerate(list(important_cells)) if
                   value_is_of_interest(ind, c, one_after_current_cell)), (-1, -1))
    (index_of_interest, value_of_interest) = result

    curr_cell = important_cells[0]  # value in the current cell
    if col == 0 and curr_cell == -1 and random.random() < prop_new_car:
        # Cars will appear randomly at the beginning of each column, if there is space
        return random.randint(1, max_model_speed)
    elif col == n_cols - 1:
        # Cars will disappear at the end of each column
        return -1
    elif index_of_interest == -1:
        # If no value of interest is found, the cell is empty in the next time step
        return -1
    else:
        # The car with the value of interest reaches the cell. Its value depends on the rules
        index_in_correct_order = index_of_current_cell - index_of_interest
        # Index of current cell, put back to correct order
        cells_to_consider = curr_lane[index_in_correct_order + 1:]
        # To define new value, consider the cells ahead
        try:
            gap_size = list(cells_to_consider).index(next(v for v in cells_to_consider if v != -1))
            # Gap size is the space until the next cell with a car in it
        except StopIteration:
            gap_size = max_model_speed + 1
            # If no car within radius is found, the gap is wider than max_model_speed and thus irrelevant
        if gap_size <= value_of_interest:
            return_value = gap_size
            # A car within radius will break out a car behind it
        elif value_of_interest < max_model_speed:
            return_value = value_of_interest + 1
            # Speed accelerates if no car is within radius and max_speed is not yet reached
        else:
            return_value = max_model_speed
            # Speed is never faster than mox_model_speed
        if random.random() < dawning_factor:
            return return_value - 1
        else:
            return return_value


def plot2d(ca, timestep=None, title=''):
    cmap = plt.get_cmap('viridis')
    plt.title(title)
    if timestep is not None:
        data = ca[timestep]
    else:
        data = ca[-1]
    plt.figure()
    plt.imshow(data, interpolation='none', cmap=cmap)


def plot2d_animate(ca, title=''):
    cmap = plt.get_cmap('viridis')
    fig = plt.figure()
    plt.title(title)
    im = plt.imshow(ca[0], animated=True, cmap=cmap)
    i = {'index': 0}
    def updatefig(*args):
        i['index'] += 1
        if i['index'] == len(ca):
            i['index'] = 0
        im.set_array(ca[i['index']])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, interval=2000, blit=True)
    plt.show()
