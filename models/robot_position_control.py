from dynsys_framework.execution_helpers.model_executor_parameters import ModelExecutorParameters
import numpy as np
_D = "d_"
POS_COMMAND = "pos_cmd"
POS_STABLE = "pos_stab"


def model(mep: ModelExecutorParameters, input_displacement, stable_pos_ref=POS_STABLE, set_constant_pos_ref=None, pos_command_k=5.):

    mep.add(_D + POS_COMMAND, (POS_COMMAND, input_displacement, stable_pos_ref), get_d_pos_command(pos_command_k),
            default_initial_values={POS_COMMAND: np.zeros((1, 12))})

    if set_constant_pos_ref is not None:
        mep.add(stable_pos_ref, (), lambda: set_constant_pos_ref)


def get_d_pos_command(k):
    def dyn(pos_cmd, pos_displ, pos_ref):
        return (np.clip(pos_ref + np.tanh(pos_displ) * 400, a_min=100, a_max=900) - pos_cmd) * k
        # return ((pos_ref + pos_displ * 600) - pos_cmd) * k
    return dyn