#!/usr/bin/python3
import argparse
import experiments_vrep

parser = argparse.ArgumentParser(description="CPG augmented Internal-Models bootstrapping - Hexapod experiment run.\n"
                                             "This script bootstraps the ensemble of internal models by controlling"
                                             "the simulated hexapod robot. The robot is tasked to reach given point"
                                             "in the plane, but to do so it must first learn to move its body. Later"
                                             "the damage is introduced, which the robot must compensate.\n\n"
                                             "Each execution of the script creates "
                                             "hexapod_goal_reaching_<experiment_run_tag>_<run_count>.hdf5 file in "
                                             "results\\vrep_poc\\ directory, where <run_count> automatically increments"
                                             "with each execution that has the same <experiment_run_tag>. The consequent"
                                             "runs of the same <experiment_run_tag> then continues the experiment run."
                                 )
parser.add_argument('experiment_run_tag', help="Experiment run identifier. Each call of this script with the same"
                                               "experiment_run_tag continues the experiment run from where it was left.")
parser.add_argument('iterations', type=int, help="Number of iterations. (recommended 200000)")

if __name__ == '__main__':
    args = parser.parse_args()
    experiments_vrep.main(experiment_run_tag=args.experiment_run_tag, total_iteration_n=args.iterations)

