import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from settings import *
maxIndex = n_rows
pairIndexTStart = y = {i: 0 for i in range(n_rows)}
pairIndexDeltaT = []
random.seed(seed)

def evolve2d(cellular_automaton, timesteps, apply_rule, r=1):
    rows, cols, _ = cellular_automaton.shape
    array = np.zeros((timesteps, rows, cols, 2), dtype=cellular_automaton.dtype)
    array[0] = cellular_automaton

    def get_neighbourhood(cell_layer, row, col):
        # Neighborhood of traffic jam:
        col_indices = range(col - r, col + r + 1)
        row_indices = range(row - 1, row + 2)
        row_indices = [i for i in row_indices if 0 <= i < n_rows]
        col_indices = [i for i in col_indices if 0 <= i < n_cols]
        return cell_layer[np.ix_(row_indices, col_indices)]

    for t in range(1, timesteps):
        cell_layer = array[t - 1]  # cell_layer represents the grid at time t-1
        for row, cell_row in enumerate(cell_layer):
            for col, cell in enumerate(cell_row):
                n = get_neighbourhood(cell_layer, row, col)
                array[t][row][col] = apply_rule(n, (row, col), t)
    return array


def value_is_of_interest(index, cell, next_cell):
    # The value of interest will reach the current cell in the next time step
    car_will_reach_cell = index == cell[0]
    car_has_to_break = next_cell[0] != -1 and cell[0] > index
    return car_will_reach_cell or car_has_to_break


def traffic_jam_rule(neighborhood, c, t):
    (row, col) = c
    if row == 0:
        lane_left = None
        curr_lane = neighborhood[0, :]
        lane_right = neighborhood[1, :]
    elif row == n_rows - 1:
        lane_left = neighborhood[0, :]
        curr_lane = neighborhood[1, :]
        lane_right = None
    else:
        lane_left = neighborhood[0, :]  # lane from which a car could arrange back, using this lane
        curr_lane = neighborhood[1, :]  # the normal row is in the middle of the neighbor rows
        lane_right = neighborhood[2, :]  # lane from which a car could be overtaking, using this lane

    index_of_current_cell = col if col <= radius else radius  # index of current cell within neighborhood
    important_cells = list(curr_lane[:index_of_current_cell + 2])  # cells until one after current cell

    if col != n_cols-1:
        one_after_current_cell = important_cells.pop()  # value after the current cell
    else:
        one_after_current_cell = [-1,-1]
    important_cells.reverse()
    # Get the closest car to the cell
    result = next(((ind, c) for ind, c in enumerate(list(important_cells)) if
                   c[0] != -1), (-1, (-1, -1)))
    (index_of_interest, cell_of_interest) = result
    speed_of_interest = cell_of_interest[0]
    not_of_interest = False
    # check if the car is of interest : if the speed is enough to reach the cell
    if not value_is_of_interest(index_of_interest, cell_of_interest, one_after_current_cell):
        not_of_interest = True
    curr_cell = important_cells[0]  # value in the current cell
    if col == 0 and row == 2 and curr_cell[0] == -1 and random.random() < prop_new_car:
        # Cars will appear randomly at the beginning of each column, if there is space
        global maxIndex
        maxIndex += 1 # index of the cell that appeared
        global pairIndexTStart
        pairIndexTStart[maxIndex] = t
        return [random.randint(1, max_model_speed), maxIndex]
    elif col == n_cols - 1 and int(speed_of_interest) > index_of_interest != -1:
        # cars with enough speed will disappear
        tStart = pairIndexTStart[int(cell_of_interest[1])]
        deltaT = t - tStart
        global pairIndexDeltaT
        # we store the time they stayed in pairIndexDeltaT
        pairIndexDeltaT.append([cell_of_interest[1], deltaT])
        return [-1, -1]
    elif index_of_interest == -1 or not_of_interest: # if there's no car in the lane or if it doesn't go fast enough
        # If no value of interest is found, the cell would be empty in the next time step
        # However, from the neighboring lanes, a car could arrange back or overtake

        # Check if a car will overtake
        if lane_right is not None:
            overtake_from = list(lane_right[:index_of_current_cell])  # cells until current column in the overtaker lane
            overtaking_car = next(((ind, c) for ind, c in enumerate(list(overtake_from)) if c[0] > -1), (-1, (-1, -1)))
            relevant_cells = overtake_from[int(overtaking_car[0] + 1):int(overtaking_car[0] + overtaking_car[1][0])]
            car_needs_to_overtake = np.any([c[0] != -1 for c in relevant_cells])  # there is a car that forces another one to brake
            space_for_overtake = np.all([c[0] == -1 for c in overtake_from[:int(overtaking_car[0])]])  # Space behind car
            space_for_overtake &= np.all([c[0] == -1 for c in curr_lane[:index_of_current_cell]])  # Space in lane
            if lane_left is not None:  # If there is a left lane, check if no car is coming from there
                space_for_overtake &= np.all([c[0] == -1 for c in lane_left[:index_of_current_cell]])
            car_has_relevant_speed = overtaking_car[1][0] == index_of_current_cell - overtaking_car[0]
            if car_needs_to_overtake and space_for_overtake and car_has_relevant_speed:
                return overtaking_car[1]

        # Check if a car will arrange back
        if lane_left is not None:
            arrange_back_from = list(lane_left[:index_of_current_cell + 1])  # cells until current column in arrange-back lane
            arrange_back_from.reverse()
            arrange_back_car = next(((ind, c) for ind, c in enumerate(list(arrange_back_from)) if c[0] == ind != 0), (-1, (-1, -1)))
            space_for_arranging_back = np.all([c[0] == -1 for c in important_cells[:int(max_model_speed+1)]])  # Space in lane
            space_for_arranging_back &= np.all([c[0] == -1 for c in arrange_back_from[:int(arrange_back_car[0])]])
            if space_for_arranging_back:
                return arrange_back_car[1]

        return [-1, -1]
    else:
        # The car with the value of interest reaches the cell. Its value depends on the rules
        index_in_correct_order = index_of_current_cell - index_of_interest
        # Index of current cell, put back to correct order

        # First, check if the car can arrange back. If so, the cell will be empty
        if lane_right is not None:
            print("lane right : " +str(lane_right))
            if (index_in_correct_order + speed_of_interest - max_model_speed) > 0 :
                index_to_start_checking = index_in_correct_order + speed_of_interest - max_model_speed
            else :
                index_to_start_checking = 0
            print(index_to_start_checking)
            print(index_in_correct_order)
            print(speed_of_interest)
            cells_to_arange_back = lane_right[int(index_to_start_checking):int(index_in_correct_order + speed_of_interest+1)]
            print(lane_right[index_in_correct_order:][:int(speed_of_interest + 1)])
            if np.all([c[0] == -1 for c in cells_to_arange_back]) and speed_of_interest != 0:
                return [-1, -1]

        cells_to_consider = curr_lane[index_in_correct_order + 1:]
        # To define new value, consider the cells ahead
        try:
            gap_size = list(cells_to_consider[:, 0]).index(next(v[0] for v in cells_to_consider if int(v[0]) != -1))
            # Gap size is the space until the next cell with a car in it
            # However, if it is too small and the car can overtake, the cell will still be empty

            # The car can only overtake, if the second lane to the left is empty, too
            # Otherwise, a back arranging and an overtaking car could collide
            if speed_of_interest > gap_size and lane_left is not None:
                cells_to_overtake = lane_left[index_in_correct_order:][:int(gap_size)]
                if np.all([c[0] == -1 for c in cells_to_overtake]):
                    return [-1, -1]

        except StopIteration:
            gap_size = max_model_speed + 1
            # If no car within radius is found, the gap is wider than max_model_speed and thus irrelevant
        if gap_size <= speed_of_interest:
            return_value = gap_size
            # A car within radius will break out a car behind it
        elif speed_of_interest < max_model_speed:
            return_value = speed_of_interest + 1
            # Speed accelerates if no car is within radius and max_speed is not yet reached
        else:
            return_value = max_model_speed
            # Speed is never faster than mox_model_speed
        if random.random() < dawning_factor:
            return [return_value - 1, cell_of_interest[1]]
        else:
            return [return_value, cell_of_interest[1]]


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
    im = plt.imshow(ca[:, :, 0], animated=True, cmap=cmap)
    i = {'index': 0}
    def updatefig(*args):
        i['index'] += 1
        if i['index'] == len(ca):
            i['index'] = 0
        im.set_array(ca[i['index']])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, interval=2000, blit=True)
    plt.show()


def saveImages(ca, title=''):
    print(pairIndexDeltaT)
    newpath = "./resources"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    fig = plt.figure()
    plt.title(title)

    for i in range(len(ca)):
        plt.imshow(ca[i], vmin=-1, vmax=7)
        plt.savefig(f'./resources/{i}.png')
