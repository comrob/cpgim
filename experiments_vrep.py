from dynsys_framework.execution_helpers.model_executor_parameters import ModelExecutorParameters
from utils import experiment_helpers
import os
import numpy as np
import models as M

RESULTS_PATH = os.path.join("results", "poc_vrep")
STEP = 0.01
LITE = "_lite.hdf5"

EXP_LEARNING_HEXAPOD = "learning_hexapod"
EXP_LEARNING_HEXAPOD_MULTI_SENSOR = "learning_hexapod_mlt_sns"
EXP_HEXAPOD_GOAL_REACHING = "hexapod_goal_reaching"
EXP_TUNNING = "cpg_sync"
EXP_SYNC_REGION_SIMPLE = "sync_region_simple"


def log_callback(state):
    cmd_context = np.argmax(state[M.limit_cycle_controller_contextual.CONTEXT_COMMANDING_SELECTION])

    system_is_learning = np.sum(state[M.limit_cycle_controller_contextual.CONTEXT_LEARNING_SELECTION]) > 0
    system_is_controlling = np.sum(state[M.limit_cycle_controller_contextual.CONTEXT_CONTROL_SELECTION]) > 0
    sys_mod = ""
    if system_is_learning:
        sys_mod += "l"
    if system_is_controlling:
        sys_mod += "c"

    cmd_context_mean_q = np.log10(state[M.limit_cycle_controller_contextual.CONTEXT_MODEL_QUALITY_MEAN][0, cmd_context])
    thr_outer = np.log10(state[M.limit_cycle_controller_contextual.CONTEXT_QUALITY_OUTER_THRESHOLD_DYN])[0]
    thr_inner = np.log10(state[M.limit_cycle_controller_contextual.CONTEXT_QUALITY_INNER_THRESHOLD_DYN])[0]

    pdf = np.around(state[M.limit_cycle_controller_contextual.PROB_JOINT_SENSORYMOTOR][0, 2, cmd_context], decimals=4)

    prf = np.log10(state[M.limit_cycle_controller_contextual.REAL_PERFORMANCE_QUALITY])
    ret = "ctx[{}] q:({:.3}>{:.3}>{:.3}) pdf:{:.3} prf:({:.3}) {}".format(
        cmd_context, thr_outer, cmd_context_mean_q, thr_inner,
        pdf,
        prf,
        sys_mod
    )
    return ret


def experiment_filename(_source, _setup, _tag):
    return "{}_{}_{}".format(_source, _setup, _tag)

# TODO folder creation
if __name__ == '__main__':
    extra_tag = "_bd10"
    mode = [EXP_HEXAPOD_GOAL_REACHING]

    if EXP_HEXAPOD_GOAL_REACHING in mode:
        extra_tag = "_030722_d"

        target_experiment_tag = EXP_HEXAPOD_GOAL_REACHING + extra_tag
        num_state_centers = 8
        sensor_dim = 4

        ##
        total_iteration_n = 200000
        continue_to = 1
        # start_at = 1440
        start_at = -1
        record_jump = 100
        record_capacity = 40
        sens_window = 40

        # command_override_start = 1500
        command_override_start = 0


        trg_coords = np.asarray([50, 50])
        pos_ref = np.zeros((1, 12)) + 500
        pos_ref[:, [3, 7, 11]] = 350 # right
        pos_ref[:, [2, 6, 10]] = 380 # left

        mep = ModelExecutorParameters()
        mep.add("d_t", (), lambda: 1, default_initial_values={"t": 0})
        M.limit_cycle_controller_contextual.model(mep,
                                                  epicycle_size=64, phase_velocity=4.,
                                                  motor_dim=12, motor_segments_num=num_state_centers,
                                                  sensory_dim=sensor_dim, sensory_segments_num=num_state_centers,
                                                  context_num=10,
                                                  model_quality_lower_bound=1e0, model_quality_upper_bound=1e1,
                                                  performance_quality_upper_bound=2e1,
                                                  perturbation_probability=0.01,
                                                  # perturbation_probability=0.0,
                                                  is_model_learned=True,
                                                  is_control_learned=True, is_model_lr_external=True,
                                                  dynamic_outer_threshold=False,
                                                  regularization_name=M.hexapod_commands_regularization.MOTOR_EXPECTED_REGULARIZATION
                                                  )
        M.hexapod_commands_regularization.model(mep,
                                                optimized_u_name=M.limit_cycle_controller_contextual.MOTOR_EXPECTED,
                                                base_u_name=M.limit_cycle_controller_contextual.MOTOR_CONTEXT,
                                                selection_vector_name=M.limit_cycle_controller_contextual.CONTEXT_CONTROL_SELECTION
                                                )
        M.robot_position_control.model(mep, M.limit_cycle_controller_contextual.MOTOR_COMMAND,
                                       set_constant_pos_ref=pos_ref, pos_command_k=30.)
        M.robot_goal_differential_control.model(mep)

        # Target coords
        mep.add(M.robot_goal_differential_control.INT_ON, M.limit_cycle_controller_contextual.CONTEXT_CONTROL_SELECTION, lambda _sel: np.sum(_sel) > 0.5)
        mep.add(M.robot_goal_differential_control.TARGET_POSITION, (), lambda: trg_coords)
        mep.add(M.robot_goal_differential_control.CURRENT_POSITION,
                (experiment_helpers.HEAD_ACC_SENSE, experiment_helpers.HEAD_CHNG_SENSE),
                lambda vel, ang: np.asarray([vel[0, 9, 0], vel[0, 10, 0], ang[0, 8, 0]])
                )

        # Mapping task-space into motor-space
        mep.add("y_goal_setup",
                M.robot_goal_differential_control.COMMAND,
                M.robot_goal_differential_control.get_command_transformation(num_state_centers))

        mep.add(M.limit_cycle_controller_contextual.SENSORY_REFERENCE, "y_goal_setup", lambda y_goal: y_goal[:, :, :, :, 0])
        mep.add(M.limit_cycle_controller_contextual.SENSORY_REFERENCE_MASK, "y_goal_setup", lambda y_goal: y_goal[:, :, :, :, 1])
        mep.add(M.limit_cycle_controller_contextual.SENSORY_REFERENCE_MINMAX_MASK, "y_goal_setup", lambda y_goal: y_goal[:, :, :, :, 2])

        # mep.add(M.limit_cycle_controller_contextual.MOTOR_LEARNING_RATE, "t", lambda _t: lr_mod[int(_t / STEP)] / 10.)
        # syncing - NO SYNC
        mep.add(M.limit_cycle_controller_contextual.MOTOR_PERTURBATION, (), lambda: 0)
        mep.add(M.limit_cycle_controller_contextual.STATE_PERTURBATION, (), lambda: 0)

        # Connecting controller with robot:
        def override_leg_command(_command, _t):
            _ret = np.zeros(_command.shape)
            _ret += _command
            if command_override_start < _t:
                # _ret[:, 0] = 500
                # _ret[:, 2] = 400
                # _ret[:, 1] = 500
                # _ret[:, 3] = 400
                _ret[:, 8] = 500
                # _ret[:, 10] = 400

            return _ret

        mep.add(experiment_helpers.DEFAULT_POS_COMMAND, (M.robot_position_control.POS_COMMAND, "t"), override_leg_command)


        mep.add(M.limit_cycle_controller_contextual.SENSORY_INPUT,
                (experiment_helpers.HEAD_ACC_SENSE, experiment_helpers.HEAD_CHNG_SENSE),
                lambda vel, ang: np.asarray([
                    vel[0, 0, 0] * 10000.,
                    ang[0, 0, 0] * 1000.,
                    ang[0, 1, 0] * 1000.,
                    ang[0, 2, 0] * 1000.]).reshape((1, -1)))

        if continue_to == 0:
            # learning rate
            lr_mod = np.zeros((total_iteration_n + 1,))
            lr_mod[10000:] = 10.
            mep.add(M.limit_cycle_controller_contextual.MODEL_LEARNING_RATE, "t", lambda _t: lr_mod[int(_t / STEP)])

            # Creating and running the model
            binders = {}
            sensory_providers = experiment_helpers.prepare_vrep_binders(binders, sensory_mean_window=sens_window)
            mod_exec = experiment_helpers.build_executor(mep, STEP, binders=binders)
            experiment_helpers.run_vrep_executor(mod_exec, total_iteration_n, sensory_providers, RESULTS_PATH,
                                                 target_experiment_tag + "_0", record_jump=record_jump, record_capacity=record_capacity,
                                                 pos_command=experiment_helpers.DEFAULT_POS_COMMAND,
                                                 callback_printer=log_callback)
        else:
            # learning rate
            lr_mod = np.zeros((total_iteration_n + 1,)) + 10.
            mep.add(M.limit_cycle_controller_contextual.MODEL_LEARNING_RATE, "t", lambda _t: lr_mod[int(_t / STEP)])


            # Creating and running the model
            binders = {}
            sensory_providers = experiment_helpers.prepare_vrep_binders(binders, sensory_mean_window=sens_window)
            mod_exec = experiment_helpers.build_executor(mep, STEP, binders=binders,
                                                         source_record_path=os.path.join(
                                                             RESULTS_PATH,
                                                             target_experiment_tag + "_{}.hdf5".format(continue_to - 1)),
                                                         from_iteration=start_at
                                                         )
            experiment_helpers.run_vrep_executor(mod_exec, total_iteration_n, sensory_providers, RESULTS_PATH,
                                                 target_experiment_tag + "_{}".format(continue_to), record_jump=record_jump,
                                                 pos_command=experiment_helpers.DEFAULT_POS_COMMAND,
                                                 callback_printer=log_callback, record_capacity=record_capacity)
