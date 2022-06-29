import numpy as np

import utils.robot_utils.provider_helpers
from dynsys_framework.dynamic_system.model import Model
from dynsys_framework.dynamic_system.model_executor import ModelExecutor
from dynsys_framework.dynamic_system.solvers import euler_solver,runge_kutta4_solver
from dynsys_framework.execution_helpers.model_executor_parameters import ModelExecutorParameters
from utils import records as R
import os
import time

from utils.robot_utils import commander as C
from utils.robot_utils.updatable_provider import UpdatableProvider, ProviderContainer
from utils.robot_utils.sensory_providers import AccelerationProvider, YAxisAccTrqProvider, GenericProvider, OrientedPositionDisplacementRate, HeadingChangeRate
from utils.robot_utils import robot_utils as RU

# try:/
import robot.hexapod_sim.RobotHAL as rob
# except:
#     print("WARNING: robot package is not present. No VREP experiments can be done.")
#     rob = None

from concurrent.futures import ThreadPoolExecutor

DEFAULT_POS_COMMAND = "pos"
ACCELERATION_SENSE = "s_acc"
HEAD_ACC_SENSE = "s_hac"
HEAD_CHNG_SENSE = "s_hch"
JOINT_TORQUES = "s_jtq"
BODY_TORQUES = "s_btq"


def get_pulses(pulse_sets, signal_iters):
    """

    :param pulse_sets: [(array_value, start_iter, iters), ...]
    :param signal_iters:
    :return:
    """
    dim = pulse_sets[0][0].shape[-1]
    pulses = np.zeros((signal_iters, 1, dim))
    for pulse_param in pulse_sets:
        value, start, length = pulse_param
        pulses[start:start+length, 0, :] += value
    return pulses


def build_executor(mep: ModelExecutorParameters, step, source_record_path=None, exclude_source_variables=["t"],
                   from_iteration=-1, binders=None):
    if binders is None:
        binders = dict()
    model = Model(mep.generate_function_dictionary(), mep.generate_processes())
    model_exec = ModelExecutor(model, runge_kutta4_solver(step), binders)
    # model_exec = ModelExecutor(model, euler_solver(step), binders)
    initializations = mep.generate_initializations()
    if source_record_path is not None:
        param_record = R.load_records(source_record_path)[0]
        R.load_init_state_from_record(param_record, initializations,
                                      variables_to_set=[k for k in initializations if k not in exclude_source_variables],
                                      from_iteration=from_iteration)
        del param_record
    model_exec.initialize_integrating_variables(initializations)
    return model_exec


def run_executor(model_exec: ModelExecutor, total_iteration_n, results_path, target_experiment_name, record_jump=1,
                 external_record_container=None):
    if external_record_container is None:
        record = R.init_record(model_exec)
    else:
        record = external_record_container
    start_t = time.time()
    start_t_l = start_t
    for i in range(total_iteration_n):
        model_exec()
        state = model_exec.get_last_state()
        if i % record_jump == 0:
            R.append_state(record, state)
        if i % 1000 == 0:
            end_t_l = time.time()
            print("iter: {}, t: {}, cmp_time: {}".format(i, state["t"], end_t_l-start_t_l))
            start_t_l = end_t_l

    end_t = time.time()
    print("time: {}".format(end_t - start_t))
    record = R.numpyfy_record(record)
    R.save_records(os.path.join(results_path, target_experiment_name + ".hdf5"), record)
    R.save_records(os.path.join(results_path, target_experiment_name + "_lite.hdf5"), R.crop_record(record,
                                                                                                    total_iteration_n - 100,
                                                                                                    total_iteration_n - 1))


def prepare_vrep_binders(binders: dict, sensory_mean_window: int = 20):

    # def binder_s_real(binders: dict, provider_container: ProviderContainer):
    #     provider = YAxisAccTrqProvider(first_empty=True)
    #     provider.set_smoothing_processor(utils.robot_utils.provider_helpers.WindowMean(np.zeros(
    #         tuple([sensory_mean_window] + list(YAxisAccTrqProvider.EXPECTED_SHAPE))
    #     )))
    #     provider_container.add_provider(provider, ACCELERATION_SENSE)
    #     binders[ACCELERATION_SENSE] = provider

    def binder_fwr_acc(binders: dict, provider_container: ProviderContainer):
        provider = OrientedPositionDisplacementRate(first_empty=True)
        provider.set_smoothing_processor(utils.robot_utils.provider_helpers.WindowMean(np.zeros(
            (sensory_mean_window, 1)
        )))
        provider_container.add_provider(provider, HEAD_ACC_SENSE)
        binders[HEAD_ACC_SENSE] = provider

    def binder_orient_displ(binders: dict, provider_container: ProviderContainer):
        provider = HeadingChangeRate(first_empty=True)
        provider.set_smoothing_processor(utils.robot_utils.provider_helpers.WindowMean(np.zeros(
            (sensory_mean_window, 3)
        )))
        provider_container.add_provider(provider, HEAD_CHNG_SENSE)
        binders[HEAD_CHNG_SENSE] = provider

    # def binder_joint_force(binders: dict, provider_container: ProviderContainer):
    #     provider = GenericProvider(expected_shape=(1,1,1),first_empty=True)
    #     provider.set_smoothing_processor(utils.robot_utils.provider_helpers.WindowMean(np.zeros(
    #         (sensory_mean_window, 1)
    #     )))
    #     provider_container.add_provider(provider, JOINT_TORQUES)
    #     binders[JOINT_TORQUES] = provider

    # def body_torque_binder(binders: dict, provider_container: ProviderContainer):
    #     provider = GenericProvider(expected_shape=(1,3,1),first_empty=True)
    #     provider_container.add_provider(provider, BODY_TORQUES)
    #     binders[BODY_TORQUES] = provider

    def debug_binder(binders: dict, provider_container: ProviderContainer):
        provider = AccelerationProvider(first_empty=True)
        provider_container.add_provider(provider, "debug")
        binders["debug"] = provider

    def debug_quaternion_binder(binders: dict, provider_container: ProviderContainer):
        provider = GenericProvider(expected_shape=(1,4,1),first_empty=True)
        provider_container.add_provider(provider, "debug_q")
        binders["debug_q"] = provider

    def debug_position_binder(binders: dict, provider_container: ProviderContainer):
        provider = GenericProvider(expected_shape=(1,3,1),first_empty=True)
        provider_container.add_provider(provider, "debug_loc")
        binders["debug_loc"] = provider

    sensory_providers = ProviderContainer()

    # binder_s_real(binders, sensory_providers)
    binder_fwr_acc(binders, sensory_providers)
    binder_orient_displ(binders, sensory_providers)
    # body_torque_binder(binders, sensory_providers)
    # binder_joint_force(binders, sensory_providers)

    debug_binder(binders, sensory_providers)
    debug_quaternion_binder(binders, sensory_providers)
    debug_position_binder(binders, sensory_providers)

    return sensory_providers


def run_vrep_executor(model_exec: ModelExecutor, total_iteration_n,
                        sensory_providers: ProviderContainer, results_path, target_experiment_name, record_jump=1,
                        pos_command=DEFAULT_POS_COMMAND, sleep_time=0.01,
                        record_capacity: int = 1000, record_queue_wait: float = 10., record_consumer_wait: float =1.,
                        callback_printer: callable = None, log_rate: int = 1000
                      ):
    # STARTING ROBOT
    robot = rob.RobotHAL(simulation_step=1)
    # record = R.init_record(model_exec)
    recorder = R.Recorder(results_path, target_experiment_name, record_capacity, lambda: R.init_record(model_exec))

    # FINISHING BINDERS (Inputs)
    # get_acceleration = robot.robot.get_imu_data
    get_orientation = robot.get_robot_orientation
    get_position = robot.get_robot_position
    # get_joint_trqs = robot.robot.get_all_joint_torques

    # get_quaternion = lambda : [0,0,0,0]
    get_quaternion = robot.get_robot_quaternion

    # def forward_downward_acc():
    #     _a = get_acceleration()
    #     if _a is not None:
    #         return [_a[0], (_a[2] + 9.81)*15]
    #     return None

    def heading_acc():
        return get_position(), get_quaternion()
    #
    # def get_joint_forces():
    #     return sum(get_joint_trqs())

    # sensory_providers.get_provider(ACCELERATION_SENSE).set_data_getter(get_acceleration)
    sensory_providers.get_provider(HEAD_ACC_SENSE).set_data_getter(heading_acc)
    sensory_providers.get_provider(HEAD_CHNG_SENSE).set_data_getter(get_quaternion)
    # sensory_providers.get_provider(BODY_TORQUES).set_data_getter(get_acceleration)

    # sensory_providers.get_provider(JOINT_TORQUES).set_data_getter(get_joint_forces)

    sensory_providers.get_provider("debug").set_data_getter(get_orientation)
    sensory_providers.get_provider("debug_q").set_data_getter(get_quaternion)
    sensory_providers.get_provider("debug_loc").set_data_getter(get_position)

    # SETTING UP THE COMMANDERS (Outputs)
    pose_commander = C.Commander(robot.set_servo_position, C.get_filter_similar_commands(0.01))

    for jid in range(0, 19):
        robot.set_joint_torque(jid, 2.5)
    start_t = time.time()
    start_t_l = start_t
    sleep_time = 0.01

    for i in range(500):
        sensory_providers.update()
        time.sleep(sleep_time)

    inner_lens = [0]
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(R.recorder_consumer, recorder, record_queue_wait, record_consumer_wait, ignore_gradients=True)
        error_flag = False
        try:
            for i in range(total_iteration_n):
                inner_time_start = time.time()
                # Providers update
                sensory_providers.update()

                # Integration step
                model_exec()
                state = model_exec.get_last_state()

                # FEEDING COMMAND
                # Pose command feeding
                pose_command_input = RU.get_command(state[pos_command], 700)
                pose_commander(RU.ticks_to_angle(pose_command_input))
                if i % record_jump == 0:
                    # R.append_state(record, state)
                    recorder.append(state)
                if i % log_rate == 0:
                    end_t_l = time.time()
                    if callback_printer is not None:
                        cll_string = callback_printer(state)
                        print("iter:{}. {:.3}/<{:.4}>s: {}".format(i, end_t_l - start_t_l, np.average(inner_lens), cll_string))

                    else:
                        print("iter:{}. {:.3}/<{:.4}>s.".format(i, end_t_l - start_t_l, np.average(inner_lens)))

                    start_t_l = end_t_l
                inner_time_end = time.time()
                inner_len = (inner_time_end - inner_time_start)
                inner_lens.append(inner_len)
                time.sleep(max(sleep_time - inner_len, 0))
        except Exception as e:
            error_flag = True
            raise e
        finally:
            recorder.flush_and_stop()
            # recorder.record_queue.join()
            while not recorder.record_queue.empty():
                print("sleeping cuz: {}".format(recorder.record_queue.qsize()))
                time.sleep(2)
            if not error_flag:
                recorder.merge(delete_iteration_record_fragments=True)