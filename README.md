# AMS_Traffic_Jam
Project of Advanced Modeling &amp; Simulation in TU Wien 2021

## Preparation

### Install all required packages
```shell script
pip install -r requirements.txt
```

### Settings
Use `settings.py` to set the desired settings of the simulation.
To run the simulation, run
```shell
python main.py
```

## Run

Calling `tjf.evolve2d(...)` computes all time steps of the cellular automaton.
`tjf.parametrize_goal_time(...)` runs the calibration process.

## Results

Visualizations can be found in the folder `resources/` after running the simulation.