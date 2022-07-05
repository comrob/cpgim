import argparse
import nn_journal_vis
import os

parser = argparse.ArgumentParser(description="CPG augmented Internal-Models bootstrapping - Hexapod experiment visualisation.\n"
                                 "This script visualises results from the goal reaching experiment."
                                 "The outputs are stored in 'results/nn/hexapod_goal_reaching_<experiment_run_tag>' directory."
                                 )
parser.add_argument('experiment_run_tag', help="Experiment run identifier.")
parser.add_argument('--show_plots', default=False, type=bool, help="If True, plots will be displayed.")
parser.add_argument('--plot_format', default='png', type=str, help="Format in which plots will be stored. 'png' or 'pdf'")


if __name__ == '__main__':
    args = parser.parse_args()
    nn_journal_vis.main(experiment_run_tag=args.experiment_run_tag, show_plots=args.show_plots, type_psf=args.plot_format)

