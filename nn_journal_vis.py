from models.visuals import limit_cycle_controller, limit_cycle_controller_contextual, robot_position_control, robot_goal_differential_control
from models.visuals import nn_journal_visuals
from experiments_vrep import RESULTS_PATH
import os
from utils import records as R
from utils import experiment_helpers as EH
import matplotlib.pyplot as plt
from experiments_vrep import EXP_HEXAPOD_GOAL_REACHING
import numpy as np
# plt.xkcd(scale=1, length=100, randomness=1)

STORE_PDF = "pdf"
STORE_PNG = "png"


class Counter:
    def __init__(self):
        self._i = 0

    def __call__(self, *args, **kwargs):
        self._i += 1
        return self._i


def main(experiment_run_tag, show_plots=True, type_psf=STORE_PNG, paralysis_start=1500, learning_wait=100):
    experiment_tag = experiment_run_tag

    mode = [1]  # cpg - embedding visualisation

    run_name = EXP_HEXAPOD_GOAL_REACHING + "_" +experiment_tag
    file_path = os.path.join(RESULTS_PATH, run_name + ".hdf5")
    output_path = os.path.join("results", "nn")
    output_path = os.path.join(output_path, run_name)


    if not os.path.exists(file_path):
        EH.merge_run_records(RESULTS_PATH, run_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # NN
    if 1 in mode:

        damage_iter = paralysis_start
        learning_iter = learning_wait

        record = R.load_records(file_path)[0]

        R.print_record_shapes(record)
        ctr = Counter()

        # Predraw analysis
        num_of_contexts = record["W_mod"].shape[-1]
        num_of_active_contexts = 0
        lrn_ctr_cmd_durs = []
        for i in range(num_of_contexts):
            lrn_durs, ctr_durs, cmd_durs = limit_cycle_controller_contextual.get_context_management_durations(record,
                                                                                                              ctx=i)
            # print((lrn_durs, ctr_durs, cmd_durs))
            lrn_ctr_cmd_durs.append((lrn_durs, ctr_durs, cmd_durs))
            if len(cmd_durs) > 0:
                num_of_active_contexts += 1

        # Drawing plots
        """
        Model parameters comparison
        """
        #
        # cmap='RdGy'
        # cmap='PuOr'
        # cmap='PiYG'
        cmap = 'RdBu'
        # cmap='RdBu'
        # cmap='PRGn'
        plt.rcParams["figure.figsize"] = (12, 12)
        for i in range(num_of_active_contexts):
            nn_journal_visuals.multi_io_matricies(record, plt.figure(ctr()), ctx=i, cmap=cmap,
                                                  sensor_modewise_norming=True,
                                                  override_title="Weights of context {}".format(i))
            plt.savefig(os.path.join(output_path, "W_ctx{}.{}".format(i,type_psf)), bbox_inches='tight')

        """
        Gait pattern comparison
        """
        for i in range(num_of_active_contexts):
            ctr_durs = lrn_ctr_cmd_durs[i][1]
            if len(ctr_durs) == 0:
                continue
            ctr_start = ctr_durs[0][0][0]
            ctr_end = ctr_durs[0][1][0]
            interval = ((ctr_start + ctr_end) // 2, ctr_end)
            plt.rcParams["figure.figsize"] = (8, 6)
            nn_journal_visuals.control_stat_pretty(record, plt.figure(ctr()), ctx=i,
                                                   title="Context {}'s average gait during t={}".format(i, interval),
                                                   interval=interval,
                                                   show_colorbar=True)
            plt.savefig(os.path.join(output_path, "avg_ctx_{}.{}".format(i,type_psf)), bbox_inches='tight')

            plt.rcParams["figure.figsize"] = (8, 6)
            nn_journal_visuals.control_stat_pretty(record, plt.figure(ctr()), ctx=i, stat=lambda x: np.std(x, axis=0),
                                                   title="Context {}'s gait deviation during t={}".format(i, interval),
                                                   interval=interval,
                                                   show_colorbar=True)
            plt.savefig(os.path.join(output_path, "std_ctx_{}.{}".format(i,type_psf)), bbox_inches='tight')

        """
        Waypoint navigation
        """

        plt.rcParams["figure.figsize"] = (5, 5)
        nn_journal_visuals.waypoint_navigation(record, plt.figure(ctr()), damage_iter=damage_iter, title=None)
        plt.savefig(os.path.join(output_path, "navigation.{}".format(type_psf)), bbox_inches='tight')
        #
        """
        Sensor reference and efferent evolution
        """
        plt.rcParams["figure.figsize"] = (15, 6)
        nn_journal_visuals.sensory_ref_clearance(record, plt.figure(ctr()), sensors=(0, 3), damage_iter=damage_iter)
        plt.savefig(os.path.join(output_path, "y_ref_clearance.{}".format(type_psf)), bbox_inches='tight')

        """
        Estimation before after learning comparison
        """
        if record["t"][-1] > damage_iter:
            plt.rcParams["figure.figsize"] = (12, 8)
            nn_journal_visuals.sensory_estimation_single(record, plt.figure(ctr()), ctx=1, sensors=(0, 1, 2, 3),
                                                         interval=(damage_iter - 100, damage_iter + 100),
                                                         damage_iter=damage_iter,
                                                         learning_start=learning_iter, show_y_labels=True,
                                                         title="Damage occurs during context 1 Controlling",
                                                         stat_view=False)
            plt.savefig(os.path.join(output_path, "estimation_evol_damage.{}".format(type_psf)), bbox_inches='tight')

        for i in range(1, num_of_active_contexts):
            lrn_durs = lrn_ctr_cmd_durs[i][0]
            lrn_start = lrn_durs[0][0][0]
            interval = (max(lrn_start - 100, 0), min(lrn_durs[0][1][0], lrn_start + 100))
            plt.rcParams["figure.figsize"] = (12, 8)
            nn_journal_visuals.sensory_estimation_double(record, plt.figure(ctr()), ctx_previous=i - 1, ctx_current=i,
                                                         sensors=(0, 1, 2, 3),
                                                         interval=interval, damage_iter=damage_iter,
                                                         learning_start=learning_iter, show_y_labels=True,
                                                         title="Estimation during learning - context {}".format(i),
                                                         stat_view=True)
            plt.savefig(os.path.join(output_path, "estimation_evol_{}.{}".format(i, type_psf)), bbox_inches='tight')

        if show_plots:
            plt.show()


if __name__ == '__main__':
    main(experiment_run_tag="050722_a")
