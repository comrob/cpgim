from models.visuals import limit_cycle_controller, limit_cycle_controller_contextual, robot_position_control, robot_goal_differential_control
from experiments_vrep import RESULTS_PATH
import os
from utils import records as EH
import matplotlib.pyplot as plt


class Counter:
    def __init__(self):
        self._i = 0

    def __call__(self, *args, **kwargs):
        self._i += 1
        return self._i


root_folder = RESULTS_PATH
# file_name = "learning_hexapod_h10_4.hdf5"
# file_name = "learning_hexapod_aa10_1.hdf5"
# file_name = "learning_hexapod_ah10_0.hdf5"
# file_name = "learning_hexapod_mlt_sns_bd10_0.hdf5"
# file_name = "hexapod_goal_reaching_ab_1a.hdf5"
# file_name = "hexapod_goal_reaching_ab.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_zero.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_5.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq5_str10.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq420_str10.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq200_str10.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq133_str10.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq266_str10.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq400_str10.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq420_str10.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq533_str10.hdf5"
# file_name = "hexapod_goal_reaching_ab_cpg_sync_frq666_str10.hdf5"
# file_name = "hexapod_goal_reaching_300522_c_1.hdf5"
# file_name = "hexapod_goal_reaching_010722_c_1.hdf5"
# file_name = "hexapod_goal_reaching_020722_a_0.hdf5"
# file_name = "hexapod_goal_reaching_020722_b_0.hdf5"
# file_name = "hexapod_goal_reaching_020722_c_0.hdf5"
# file_name = "hexapod_goal_reaching_020722_d_0.hdf5"
file_name = "hexapod_goal_reaching_030722_d_0.hdf5"

mode = [1]
# mode = [2]
# mode = [3]
# mode += [11]


global_start = 0
global_end = -1

# global_start = 2477
# global_end = 2760

# global_start = 10000
# global_start = 80000
# global_end = 2000
# global_end = 5000
# global_end = 10000


if 1 in mode:

    start = global_start
    end = global_end

    record = EH.load_records(os.path.join(RESULTS_PATH, file_name))[0]
    record = EH.crop_record(record, start, end)

    EH.print_record_shapes(record)
    ctr = Counter()

    # limit_cycle_controller.state_evolution(record, plt.figure(ctr()))
    # limit_cycle_controller.epicycle_evolution(record,  plt.figure(ctr()))

    limit_cycle_controller_contextual.context_control_evol(record, plt.figure(ctr()))
    # limit_cycle_controller_contextual.context_quality_evol(record, plt.figure(ctr()), log=False)
    # limit_cycle_controller_contextual.motor_context_convergence(record, plt.figure(ctr()), squared_norm=True)
    # limit_cycle_controller_contextual.multi_control_outputs(record, plt.figure(ctr()))
    #
    # limit_cycle_controller.y_mem_evolution(record, plt.figure(ctr()), sensor_m=0, size=2)
    # limit_cycle_controller.y_mem_evolution(record, plt.figure(ctr()), sensor_m=1, size=2)
    # limit_cycle_controller.y_mem_evolution(record, plt.figure(ctr()), sensor_m=2, size=2)
    # limit_cycle_controller.y_mem_evolution(record, plt.figure(ctr()), sensor_m=3, size=2)
    # limit_cycle_controller.y_mem_evolution(record, plt.figure(ctr()), sensor_m=1, size=4)
    # limit_cycle_controller.y_mem_evolution(record, plt.figure(ctr()), sensor_m=2, size=4)
    #
    #
    # limit_cycle_controller.u_mem_evolution(record, plt.figure(ctr()), motor_n=0, size=2)
    # limit_cycle_controller.u_mem_evolution(record, plt.figure(ctr()), motor_n=1, size=4)
    # limit_cycle_controller.u_mem_evolution(record, plt.figure(ctr()), motor_n=2, size=4)
    # robot_position_control.position_command_evolution(record, plt.figure(ctr()))
    # limit_cycle_controller_contextual.command_evolution(record, plt.figure(ctr()))

    # robot_position_control.debug(record, plt.figure(ctr()))
    # robot_position_control.debug_q(record, plt.figure(ctr()))
    # robot_position_control.heading_acceleration(record, plt.figure(ctr()))
    # robot_position_control.provider_xy_position(record, plt.figure(ctr()), vec_step=1)
    # robot_position_control.heading_change_rate(record, plt.figure(ctr()))
    # robot_position_control.heading_acceleration(record, plt.figure(ctr()))
    # robot_position_control.s_btq(record, plt.figure(ctr()))

    # robot_position_control.regularization_global_budget(record, plt.figure(ctr()))
    # robot_position_control.regularization_u_variance(record, plt.figure(ctr()))
    # robot_position_control.regularization_contalateral_symmetry(record, plt.figure(ctr()))
    # robot_position_control.regularization_neighbour_closeness(record, plt.figure(ctr()))
    # #
    # limit_cycle_controller_contextual.joint_motorsensory_distribution(record, plt.figure(ctr()))
    # limit_cycle_controller_contextual.debug_system_performance_error(record, plt.figure(ctr()))

    # limit_cycle_controller_contextual.free_energy_motor_error_comparison(record, plt.figure(ctr()))
    # limit_cycle_controller_contextual.performance_quality_error(record, plt.figure(ctr()), log_it=True)
    #
    # robot_goal_differential_control.diff_control_evol(record, plt.figure(ctr()))
    # robot_goal_differential_control.pid_evol(record, plt.figure(ctr()))
    for i in [0, 1]:
        limit_cycle_controller_contextual.model_context_parameter_convergence(record, plt.figure(ctr()), context=i)

        limit_cycle_controller_contextual.multi_io_matricies(record, plt.figure(ctr()), ctx=i, sensor_modewise_norming=True)

        # limit_cycle_controller_contextual.weight_matrix_analysis(record, plt.figure(ctr()),sensor_m=0, motor_n=0, ctx=i)
        # limit_cycle_controller_contextual.weight_matrix_analysis(record, plt.figure(ctr()),sensor_m=0, motor_n=1, ctx=i)
        # limit_cycle_controller_contextual.weight_matrix_analysis(record, plt.figure(ctr()),sensor_m=1, motor_n=1, ctx=i)
        # limit_cycle_controller_contextual.weight_matrix_analysis(record, plt.figure(ctr()),sensor_m=2, motor_n=2, ctx=i)
        # limit_cycle_controller_contextual.motor_evolution(record, plt.figure(ctr()), ((0, 0),), ctx=i)
        # limit_cycle_controller_contextual.bias_learning(record, plt.figure(ctr()),sensor_m=0, phase_d=0, ctx=i)
        # limit_cycle_controller_contextual.bias_learning(record, plt.figure(ctr()),sensor_m=1, phase_d=0, ctx=i)
        # limit_cycle_controller_contextual.bias_learning(record, plt.figure(ctr()),sensor_m=2, phase_d=0, ctx=i)
        # limit_cycle_controller_contextual.bias_learning(record, plt.figure(ctr()),sensor_m=3, phase_d=0, ctx=i)
        # limit_cycle_controller_contextual.bias_learning(record, plt.figure(ctr()),sensor_m=0, phase_d=7, ctx=i)
        # limit_cycle_controller_contextual.bias_evolution(record, plt.figure(ctr()), ctx=i)
        # limit_cycle_controller_contextual.weight_matrix_evolution(record, plt.figure(ctr()), size=10,ctx=i)

        # limit_cycle_controller_contextual.phase_combined_sensory_evolution(record, plt.figure(ctr()), ctx=i)
        # robot_position_control.u_leg_position(record, plt.figure(ctr()), ctx=i)

        # limit_cycle_controller_contextual.sensory_posterior_variance(record, plt.figure(ctr()), ctx=i)
        # robot_position_control.regularization_leg_budget(record, plt.figure(ctr()), ctx=i)
        # limit_cycle_controller_contextual.u_dif_evolution(record, plt.figure(ctr()), ctx=i)
        #
        # lrn_durs, ctr_durs, _ = limit_cycle_controller_contextual.get_context_management_durations(record, ctx=i)
        # for m in [0, 1, 3]:
        #     for lrn_dur in lrn_durs:
        #         limit_cycle_controller_contextual.posterior_sensory_variance_detail(record, plt.figure(ctr()), t=lrn_dur[1][0],
        #                                                                             sensor_m=m, ctx=i, title_psfx=" learn")
        #     for ctr_dur in ctr_durs:
        #         limit_cycle_controller_contextual.posterior_sensory_variance_detail(record, plt.figure(ctr()), t=ctr_dur[1][0],
        #                                                                             sensor_m=m, ctx=i, title_psfx=" control")

    plt.show()

if 2 in mode:
    import models.limit_cycle_controller_contextual as M
    import models.robot_goal_differential_control as REFDF
    import numpy as np
    # difs = np.linspace(-3, 3, num)
    # frqs = [4 * i/(num/2) for i in range(num)]
    # frqs = 4 + difs
    # frqs = [np.pi/2, 3.33, 3.5, 4, 4.5, 4.83, 2*np.pi]
    frqs = [np.pi/2, 3.33, 3.5, 3.7, 4, 4.3, 4.5,  4.83,  2*np.pi]
    # frqs = [np.pi/2, 3.33, 3.5, 4, 4.5, 4.83,  2*np.pi]

    labels = [int(100*frq) for frq in frqs]
    # frqs = [133, 266, 300, 333, 400, 500, 533, 600, 666]
    record_paths = [os.path.join(RESULTS_PATH, "hexapod_goal_reaching_ab_cpg_sync_frq{}_str10.hdf5".format(frq)) for frq in labels]
    records = EH.extract_from_record_files(record_paths,
                                           [M.EPICYCLE_PHASE, M.STATE_PHASE_ESTIMATION,
                                            M.EPICYCLE_PERTURBATION, M.EPICYCLE_PHASE_ACTIVATOR, "t",
                                            M.STATE_PHASE_ACTIVATOR, M.SENSORY_EFFERENT_ESTIMATION,
                                            M.SENSORY_ESTIMATION, M.SENSORY_INPUT, M.SENSORY_REFERENCE,
                                            M.MODEL_BIAS, M.PROB_VAR_SENSORY_MOTOR, M.REAL_PERFORMANCE_QUALITY,
                                            M.CONTEXT_MODEL_QUALITY,
                                            REFDF.COMMAND, REFDF.CURRENT_COMMAND, REFDF.CURRENT_POSITION, REFDF.TARGET_POSITION
                                            ])

    EH.print_record_shapes(records[0])
    ctr = Counter()
    # for i, record in enumerate(records):
    #     robot_goal_differential_control.diff_control_evol(record, plt.figure(ctr()), title_psfx=labels[i])
    #     record = EH.crop_record(record, 100, -1)
    #     limit_cycle_controller.epicycle_syncing(record,  plt.figure(ctr()))
        # limit_cycle_controller_contextual.phase_combined_sensory_evolution(record, plt.figure(ctr()), ctx=1)

        # limit_cycle_controller.state_evolution(record, plt.figure(ctr()))
        # limit_cycle_controller_contextual.context_control_evol(record, plt.figure(ctr()))

    # for i in [1, 2]:
    #     lrn_durs, ctr_durs, _ = limit_cycle_controller_contextual.get_context_management_durations(record, ctx=i)

    # multiplots
    limit_cycle_controller_contextual.compare_performance_error_means(records, plt.figure(ctr()), labels,
                                                                      mean_interval=(-500, -1), ctx=1)


    limit_cycle_controller_contextual.compare_sensory_means(records, plt.figure(ctr()), labels, mean_interval=(-500, -1))


    limit_cycle_controller_contextual.comparison_epicycle_syncing(records, plt.figure(ctr()), np.asarray(frqs), labels,
                                                                  base_label=400)



    plt.show()

if 3 in mode:
    import utils.self_oscillator_analysis.visualisations as VIS
    import numpy as np
    num = 100
    # frqs = [4 * i / (num / 2) for i in range(1, num)]
    difs = np.linspace(-3, 3, num)
    # frqs = [4 * i/(num/2) for i in range(num)]
    frqs = 4 + difs
    labels = [int(100*frq) for frq in frqs]

    record_paths = [os.path.join(RESULTS_PATH, "sync_region_simple_fosc400_ffor{}_sfor1500_a.hdf5".format(lab)) for
                    lab in labels]
    records = EH.extract_from_record_files(record_paths, ["t", "phs", "pert"])

    EH.print_record_shapes(records[0])
    ctr = Counter()
    # multiplots
    # VIS.compare_syncing(records, plt.figure(ctr()), labels, osc_phase_label=labels[len(labels)//2])
    VIS.observed_freqs(records, plt.figure(ctr()), frequencies=[frq for frq in frqs], obs_interval=(20000, -1))

    plt.show()