
# Internal Model Learning for Limit Cycle Systems
This package contains implementation of Feed Forward Network synchronized by Central Pattern Generator (CPG-NN),
which learns limit-cycle system dynamics.
Both CPG-NN and limit-cycle system are described by differential equations which are numerically integrated by
dynsys_famework (also included in this package). 

## Installation
The software is tested on Ubuntu 20.4 and Windows 11. To run the experiments the python libraries and the simulator must be installed:

1. The software requires Python 3.8.0 with packages listed in *requirements.txt* file.
To install the requirements run:
```setup
pip install -r requirements.txt
```

2. The robot is simulated in [CoppeliaSim](https://www.coppeliarobotics.com/downloads) 
which should be installed on your machine (tested on EDU distribution version 4.3.0).


## Training the Internal Model Ensemble

>Before running the experiment in the simulator, we encourage you to try this Jupyter Notebook Demo, where we train
to control simple mass-spring-damper system. You will see how the *internal model* learns to
control a low-dimensional system. The demo will also introduce you to the *DynSys framework*
which numerically solves differential equations. 

### Experimental Setup
We run the experiment where the simulated robot learns multiple internal models, which provides the
robot with appropriate control. During the experiment the robot experiences leg damage, which must the
controller compensate.

### Running the experiment
First we prepare the simulated environment:
1. Launch CoppeliaSim
2. Open the ```scenes\plain.ttt``` scene.

You should see the hexapod robot which we are about to make move (PICREL).

Now we just run the python script
```train
python train.py <experiment_run_tag> <number_of_iterations>
```
where *model_tag* identifies the experiment run, and *number_of_iterations* is number of numerical integration steps.
> Set the *number_of_iterations* at most to 400 000.
> However such run could take 3 hours (Ubuntu20.4, Intel I7, 16GB RAM). 
> We recommend you to separate the run into shorter runs.
> Just run the 200 000 iterations twice, the results are then automatically merged.

For more options consult 
```train
python train.py -h
```

The robot should start randomly moving with its legs (motor-babbling). Usually it starts walking at 100 000th iteration.
The leg paralysis is introduced at 150 000th iteration.

## Results analysis

The entire run is stored in the hdf5 file(s) in the ```results\vrep_poc``` directory. And now we simply analyse
the stored data. Just run
```eval1
python evaluate.py <experiment_run_tag>
```
and new directory ```results\nn\<experiment_run_tag>``` will be generated with plotted data. 

For more options consult 
```train
python evaluate.py -h
```

Now we describe what each plot is supposed to show.  

## Pre-trained Models
(Will be there pretrained models?)
In *results* directory there is already included pre-trained model: *modelVDP_lite.hdf5*.
To generate evaluations run following commands:
```eval
python evaluate.py exp modelVDP one_pulse
python evaluate.py exp modelVDP perturb_and_sync
python evaluate.py exp modelVDP amplitude_grow
python evaluate.py exp modelVDP control_simple
```

## Results

The CPG-NN discovers the approximate dynamics of the limit-cycle system through perturbations.
If the CPG is not synchronized then the perturbations shift the limit-cycle system and the weights does not converge.\
![](results/_pics/learning_convergence.png)\
The result model can be used for sensory output estimation, when we send pulse into the limit-cycle system.\
![](results/_pics/one_pulse_overlap.png)\
The synchronization ability of CPG is essential, as we can see in the comparison of synchronized and unsynchronized 
CPG-NN estimations.
![](results/_pics/compare_perturbation_syncing.png)\
The learned dynamics are only approximate, if we diverge from regular state too far, the estimation error increases.
![](results/_pics/growing_amplitude.png)\
The linear model of limit-cycle can be used for finding optimal control with respect to given sensory reference. Again,
we compare synchronized and unsynchronized CPG-NN.\
![](results/_pics/period_comparison_overlap_with_control.png)
![](results/_pics/period_comparison_overlap_with_control_unsynced.png)


#### Project structure
The equations are organized into three files.
 - **models/limit_cycle_controller.py** has CPG-NN implementation.
 - **models/limit_cycle_system.py** contains Van der Pol oscillator implementation.
 - **experiments_nips.py** contain experiment setup specifications and couples the
CPG-NN with the limit cycle system.
