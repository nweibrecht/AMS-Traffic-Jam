# Settings

# 1 cell corresponds to 5 meters
# 1 time step corresponds to 1 second

n_rows = 5
what_to_block_when = [[4, 0, 10], [1, 0, 10]]  # [[row,col,t],[...]]]
# what_to_block_when = [[4, 0, 10], [1, 0, 10]]
n_cols = 50
max_model_speed = 8  # 24 m/s --> 86,4 km/h
# what_to_block_when = [[4,5,10],[1,0,10]] # [[row,col,t],[...]]]
# n_cols = 20
# max_model_speed = 7  # 35 m/s --> 126 km/h
n_time_steps = 50
n_new_cars_per_t = 1.5
prop_new_car_t0 = n_new_cars_per_t / n_rows
dawning_factor = 0.1
seed = 4
radius = max_model_speed
mean_over_all_timesteps = True
