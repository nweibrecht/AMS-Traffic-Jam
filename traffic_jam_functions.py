import os
import random
import statistics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from settings import *
maxIndex = n_rows
pairIndexTStart = y = {i: 0 for i in range(n_rows)} # Pair linking the index of the car and the time it appeared
pairIndexDeltaT = [] # Pair linking the index of the car and the time it took to to travel
pairNumberSpeed_t = [[[-1,0] for i in range(n_cols)] for j in range(n_rows)] # Pair linking the number of cars that have gone to a specific cell and the mean speed they had
pairNumberSpeed_overall = []
cells_overall = [[['' for i in range(n_cols)] for j in range(n_rows)]]
blocked_cells = dict()
random.seed(seed)
prop_new_car = prop_new_car_t0
def copyOfPairNumberSpeedT():
    newPair = []
    for row in range(n_rows):
        newPair.append([])
        for col in range(n_cols):
            newPair[row].append([pairNumberSpeed_t[row][col][0],pairNumberSpeed_t[row][col][1]])
    return newPair

def block_event(t):
    # we add a the block events that are written in what_to_block_when
    for info in what_to_block_when:
        if t == info[2] :
            blocked_cells[info[0]] = info[1]

    # formatting the cells_overall for timestep t
    cells_t = [['' for i in range(n_cols)] for j in range(n_rows)]
    n_rows_where_cars_cant_appear = 0
    for row in blocked_cells.keys():
        for i in range(int(blocked_cells[row]), n_cols):
            cells_t[row][i] = 'x'
    cells_overall.append(cells_t)


def get_prop_new_car(cars_t):
    n_rows_where_cars_cant_appear = 0
    for row in range(n_rows):
        if cars_t[row][0][0] != -1:
            n_rows_where_cars_cant_appear += 1
    global prop_new_car
    if n_rows - n_rows_where_cars_cant_appear == 0:
        prop_new_car = 1
    else :
        prop_new_car = n_new_cars_per_t / (n_rows - n_rows_where_cars_cant_appear)
    if prop_new_car > 1:
        prop_new_car = 1

def evolve2d(cellular_automaton, timesteps, apply_rule, r=1):
    rows, cols, _ = cellular_automaton.shape
    array = np.zeros((timesteps, rows, cols, 2), dtype=cellular_automaton.dtype)
    array[0] = cellular_automaton
    global pairNumberSpeed_overall
    global pairNumberSpeed_t

    pairNumberSpeed_overall.append(copyOfPairNumberSpeedT())
    def get_neighbourhood(cell_layer, row, col):
        # Neighborhood of traffic jam:
        col_indices = range(col - r, col + r + 1)
        row_indices = range(row - 2, row + 2)
        row_indices = [i for i in row_indices if 0 <= i < n_rows]
        col_indices = [i for i in col_indices if 0 <= i < n_cols]
        return cell_layer[np.ix_(row_indices, col_indices)]

    for t in range(1, timesteps):
        block_event(t)
        cell_layer = array[t - 1]  # cell_layer represents the grid at time t-1
        for row, cell_row in enumerate(cell_layer):
            for col, cell in enumerate(cell_row):
                n = get_neighbourhood(cell_layer, row, col)
                array[t][row][col] = apply_rule(n, (row, col), t)
        get_prop_new_car(array[t])

        pairNumberSpeed_overall.append(copyOfPairNumberSpeedT())
        # if we want pair NumberSpeed_speed not to take into account the other timesteps
        if not mean_over_all_timesteps:
            # reset it at everytimestep
            pairNumberSpeed_t = [[[-1,0] for i in range(n_cols)] for j in range(n_rows)]
    return array

def is_blocked(row,col) :
    blocked_rows = blocked_cells.keys()
    if row in blocked_rows and col > blocked_cells[row] : #only greater because the a car with 0 speed is placed in the =
        return True
    else :
        return False

def value_is_of_interest(index, cell, next_cell):
    # The value of interest will reach the current cell in the next time step
    car_will_reach_cell = index == cell[0]
    car_has_to_break = next_cell[0] != -1 and cell[0] > index
    return car_will_reach_cell or car_has_to_break

def add_speed_to_list(row, col, real_speed):
    # add the speed of the car to the list
    global pairNumberSpeed_t
    for i in range(real_speed):  # index_of_interest represents the real speed the car is coming with
        pairNumberSpeed_t[row][col - i] = [(pairNumberSpeed_t[row][col - i][0] * pairNumberSpeed_t[row][col - i][1] + real_speed) / (
                    pairNumberSpeed_t[row][col - i][1] + 1), pairNumberSpeed_t[row][col - i][1] + 1]


def get_lane(nh, lane_nr):
    try:
        return nh[lane_nr, :]
    except IndexError:
        return None


def traffic_jam_rule(neighborhood, c, t):
    (row, col) = c
    if row == 0:
        lane_two_left = None
        lane_left = None
        curr_lane = neighborhood[0, :]
        lane_right = get_lane(neighborhood, 1)
    elif row == 1:
        lane_two_left = None
        lane_left = neighborhood[0, :]
        curr_lane = neighborhood[1, :]
        lane_right = get_lane(neighborhood, 2)
    elif row == n_rows - 1:  # last lane, and row is at least 2
        lane_two_left = neighborhood[0, :]
        lane_left = neighborhood[1, :]
        curr_lane = neighborhood[2, :]
        lane_right = None
    else:  # middle lane, at least 2
        lane_two_left = neighborhood[0, :]
        lane_left = neighborhood[1, :]  # lane from which a car could arrange back, using this lane
        curr_lane = neighborhood[2, :]  # the normal row is in the middle of the neighbor rows
        lane_right = neighborhood[3, :]  # lane from which a car could be overtaking, using this lane

    index_of_current_cell = col if col <= radius else radius  # index of current cell within neighborhood
    important_cells = list(curr_lane[:index_of_current_cell + 2])  # cells until one after current cell

    ## check if the cell is blocked
    if is_blocked(row, col):
        return [-1, -1]
    # return the blocking object if the cell is the starting point of the blocking lane
    if row in blocked_cells and blocked_cells[row] == col :
        return [0,-1]

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
    if col == 0 and curr_cell[0] == -1 and random.random() < prop_new_car:
        # Cars will appear randomly at the beginning of each column, if there is space
        global maxIndex
        maxIndex += 1 # index of the cell that appeared
        global pairIndexTStart
        pairIndexTStart[maxIndex] = t
        speed = random.randint(1, max_model_speed)
        return [speed, maxIndex]
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
        if lane_right is not None and not is_blocked(row,col) and col != 0:
            overtake_from = list(lane_right[:index_of_current_cell+2])  # cells until current column in the overtaker lane
            overtaking_car = next(((ind, c) for ind, c in enumerate(list(overtake_from[:-1])) if c[0] == index_of_current_cell - ind and c[1] > -1), (-1, (-1, -1)))
            if overtaking_car[1][0] != 0:
                relevant_cells = overtake_from[int(overtaking_car[0] + 1):int(overtaking_car[0] + overtaking_car[1][0] + 1)]
            else:
                relevant_cells = [overtake_from[int(overtaking_car[0] + 1)]]
            car_needs_to_overtake = np.any([c[0] != -1 for c in relevant_cells])  # there is a car that forces another one to brake
            space_for_overtake = np.all([c[0] <= overtaking_car[1][0] for c in overtake_from[:int(overtaking_car[0])]])  # Space behind car
            space_for_overtake &= np.all([c[0] == -1 for c in curr_lane[:index_of_current_cell+1]])  # Space in lane
            if lane_left is not None :  # If there is a left lane, check if no car is coming from there
                space_for_overtake &= np.all([c[0] == -1 for c in lane_left[:index_of_current_cell]])
            if col == 1 and row == 1:
                print(t)
                print(overtake_from)
                print(overtaking_car)
                print(relevant_cells)
                print(car_needs_to_overtake)
                print(space_for_overtake)
            if car_needs_to_overtake and space_for_overtake:
                add_speed_to_list(row, col, int(overtaking_car[1][0]))
                if (overtaking_car[1][0] == 0):
                    overtaking_car[1][0] += 1
                return overtaking_car[1]

        # Check if a car will arrange back
        if lane_left is not None and not is_blocked(row,col):
            arrange_back_from = list(lane_left[:index_of_current_cell + 1])  # cells until current column in arrange-back lane
            arrange_back_from.reverse()
            arrange_back_car = next(((ind, c) for ind, c in enumerate(list(arrange_back_from)) if c[0] == ind != 0), (-1, (-1, -1)))
            space_for_arranging_back = np.all([c[0] == -1 for c in important_cells[:int(max_model_speed+1)]])  # Space in lane
            space_for_arranging_back &= np.all([c[0] == -1 for c in arrange_back_from[:int(arrange_back_car[0])]])
            if space_for_arranging_back:
                add_speed_to_list(row, col, arrange_back_car[0])
                return arrange_back_car[1]

        return [-1, -1]
    else:
        # The car with the value of interest reaches the cell. Its value depends on the rules
        index_in_correct_order = index_of_current_cell - index_of_interest
        # Index of current cell, put back to correct order

        # First, check if the car can arrange back. If so, the cell will be empty
        if lane_right is not None and not is_blocked(row+1,col) :
            if (index_in_correct_order + speed_of_interest - max_model_speed) > 0 :
                index_to_start_checking = index_in_correct_order + speed_of_interest - max_model_speed
            else:
                index_to_start_checking = 0
            cells_to_arange_back = lane_right[int(index_to_start_checking):int(index_in_correct_order + speed_of_interest+1)]
            space_to_arrange_back = np.all([c[0] == -1 for c in cells_to_arange_back])
            space_to_arrange_back &= np.all([c[0] == -1 for c in important_cells[:int(speed_of_interest)]])
            if space_to_arrange_back and speed_of_interest != 0:
                return [-1, -1]

        cells_to_consider = curr_lane[index_in_correct_order + 1:]
        # To define new value, consider the cells ahead
        try:
            gap_size = list(cells_to_consider[:, 0]).index(next(v[0] for v in cells_to_consider if int(v[0]) != -1))
            # Gap size is the space until the next cell with a car in it
            # However, if it is too small and the car can overtake, the cell will still be empty

            # The car can only overtake, if the second lane to the left is empty, too
            # Otherwise, a back arranging and an overtaking car could collide
            if (speed_of_interest > gap_size or speed_of_interest == 0) and lane_left is not None and not is_blocked(row-1,col) and col != 0:
                if np.all([c[0] <= speed_of_interest for c in curr_lane[:int(index_of_current_cell)]]):
                    cells_to_overtake = lane_left[:int(index_in_correct_order + speed_of_interest +1)]
                    if np.all([c[0] == -1 for c in cells_to_overtake]):
                        if lane_two_left is None:
                            return [-1, -1]
                        elif np.all([c[0] == -1 for c in lane_two_left[:int(index_in_correct_order + speed_of_interest +1)]]):
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
        add_speed_to_list(row, col, index_of_interest)
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

def getMeanSpeedPlot(ax, t, row):
    x = range(n_cols)
    if row == -1 : # mean over all rows
        y = []
        for col in range(len(pairNumberSpeed_t[0])):
            all_rows = []
            for r in range(n_rows):
                all_rows.append(pairNumberSpeed_overall[t][r][col][0])
            y.append(statistics.mean(all_rows))
    else:
        y = [pairNumberSpeed_overall[t][row][i][0] for i in range(len(pairNumberSpeed_t[0]))]

    colors = ['k','b','g','r','c','m']
    if row == -1 :
        ax.step(x, y, color=colors[row+1], label='average')

    else :
        ax.step(x, y, color=colors[row+1], label=str(row))
    ax.set_ylim(0,max_model_speed+1)
    ax.set_yticks(np.arange(0, max_model_speed+1, step=1))
    ax.set_xticks(np.arange(0, n_cols, step=2))
    ax.set_xlabel('positions of cars')
    ax.set_ylabel('average speed')
    ax.grid(axis='y')

def plot2d_animate(ca, title=''):
    cmap = plt.get_cmap('viridis')
    fig = plt.figure()
    fig.add_subplot(2,1,1)
    plt.title(title)
    im = plt.imshow(ca[:, :, 0], animated=True, cmap=cmap)
    i = {'index': 0}
    def updatefig(*args):
        i['index'] += 1
        if i['index'] == len(ca):
            i['index'] = 0
        im.set_array(ca[i['index']])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True)
    plt.show()


def create_text_tuple(i, j, t, ca, fbc, text):
    if ca[t][i][j] != -1:  # return speed of car
        return text(j, i, ca[t][i][j], ha="center", va="center")
    else:  # return 'x' or ''
        return text(j, i, fbc[i][j], ha="center", va="center", color='w')


def saveImage(ca, timestep):
    fig, axes = plt.subplots(2)
    axes[0].imshow(ca[timestep], vmin=-1, vmax=7)
    # Loop over data dimensions and create text annotations.
    formattedBlockedCells = cells_overall[timestep]
    for i in range(n_rows):
        for j in range(n_cols):
            create_text_tuple(i, j, timestep, ca, formattedBlockedCells, axes[0].text)
            # axes[0].text(j, i, formattedBlockedCells[i][j], ha="center", va="center", color="w")
    for j in range(-1,n_rows):
        getMeanSpeedPlot(axes[1], timestep, j)
    axes[1].legend(loc='lower right', framealpha=0.5, fontsize='small')

    plt.savefig(f'./resources/{timestep}.png')
    plt.close(fig)

def saveImages(ca, title=''):
    newpath = "./resources"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    #getTimePlot(axes[1])

    plt.title(title)
    for t in range(len(ca)):
        saveImage(ca,t)
