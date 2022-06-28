from dynsys_framework.functions.tensor.common import *


def counter_error_shift(alpha: float, epsilon: float):
    """
    The center (centers) is learned as position on limit cycle (inputs) that coincides with input activity (activities).
    If inputs value that coincides with activity pulse is not on centers, the counter error shift pulls inputs
    towards centers.
    :param alpha: pull strength; beware strong pull may kill the limit cycle.
    :param epsilon: neighbourhood of centre (square metric) which is not affected by this dynamics
    probably should be large enough to cover the segment of limit-cycle (inputs) selected by activity.
    :return:
    """
    def dyn(inputs, centers, activities):
        """

        :param inputs: SIGNAL TYPE, shape = (1, input_dimension, #streams)
        :param centers: WEIGHT TYPE, shape = (input_dimension, 1, #streams)
        :param activities: SIGNAL TYPE, shape = (1, 1, #streams)
        :return: uv.shape vector
        """
        diff = tadd(centers, -inputs)
        direction = xp.sign(diff)  # save directions
        # FIXME high diff can easily overflow magnitude value, there should be upper limit of diff.
        magnitude = xp.exp(xp.maximum((xp.abs(diff) - epsilon) * alpha, 0))-1  # filter out error smaller than epsilon
        return activities * direction * magnitude,
    return dyn