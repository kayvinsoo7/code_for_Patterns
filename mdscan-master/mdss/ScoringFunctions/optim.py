""" Module for optimization algorithms. """
import numpy as np
from mdss.ScoringFunctions.ScoringFunction import ScoringFunction

def bisection_q_mle(
    score_function: ScoringFunction,
    observed_sum: float,
    expectations: np.array,
    **kwargs
):
    """
    Computes the q which maximizes score (q_mle).
    Computes q for which slope dscore/dq = 0, using the fact that slope is monotonically decreasing.
    q_mle is computed via bisection.
    This works if the score, as a function of q, is concave.
    So the slope is monotonically decreasing, and q_mle is the unique value for which slope = 0.

    :param observed_sum: sum of observed binary outcomes for all i
    :param expectations: predicted outcomes for each data element i
    :return: q MLE
    """
    q_temp_min = 1e-6
    q_temp_max = 1e6
    direction = kwargs.get('direction')

    while np.abs(q_temp_max - q_temp_min) > 1e-6:
        q_temp_mid = (q_temp_min + q_temp_max) / 2

        if np.sign(score_function.q_dscore(observed_sum, expectations, q_temp_mid)) > 0:
            q_temp_min = q_temp_min + (q_temp_max - q_temp_min) / 2
        else:
            q_temp_max = q_temp_max - (q_temp_max - q_temp_min) / 2

    q = (q_temp_min + q_temp_max) / 2

    if ((direction == "positive") & (q < 1)) | ((direction == "negative") & (q > 1)):
        return 1

    return q


def bisection_q_min(
    score_function: ScoringFunction,
    observed_sum: float,
    expectations: np.array,
    penalty: float,
    q_mle: float,
    temp_min: float = 1e-6,
):
    """
    Compute q for which score = 0,
    using the fact that score is monotonically increasing for q > q_mle.
    q_max is computed via binary search.
    This works because the score, as a function of q, is concave.

    :param observed_sum: sum of observed binary outcomes for all i
    :param expectations: predicted outcomes for each data element i
    :param penalty: penalty term. should be positive
    :param q_mle: q maximum likelihood
    :return: the root on the LHS of qmle
    """
    q_temp_min = temp_min
    q_temp_max = q_mle

    while np.abs(q_temp_max - q_temp_min) > 1e-6:
        q_temp_mid = (q_temp_min + q_temp_max) / 2

        if (
            np.sign(
                score_function.score(observed_sum, expectations, penalty, q_temp_mid)
            )
            > 0
        ):
            q_temp_max = q_temp_max - (q_temp_max - q_temp_min) / 2
        else:
            q_temp_min = q_temp_min + (q_temp_max - q_temp_min) / 2

    ans = (q_temp_min + q_temp_max) / 2
    return ans


def bisection_q_max(
    score_function: ScoringFunction,
    observed_sum: float,
    expectations: np.array,
    penalty: float,
    q_mle: float,
    temp_max: float = 1e6,
):
    """
    Compute q for which score = 0,
    using the fact that score is monotonically decreasing for q > q_mle.
    q_max is computed via binary search.
    This works because the score, as a function of q, is concave.

    :param observed_sum: sum of observed binary outcomes for all i
    :param expectations: predicted outcomes for each data element i
    :param penalty: penalty term. should be positive
    :param q_mle: q maximum likelihood
    :return: the root on the RHS of qmle
    """
    q_temp_min = q_mle
    q_temp_max = temp_max

    while np.abs(q_temp_max - q_temp_min) > 1e-6:
        q_temp_mid = (q_temp_min + q_temp_max) / 2

        if (
            np.sign(
                score_function.score(observed_sum, expectations, penalty, q_temp_mid)
            )
            > 0
        ):
            q_temp_min = q_temp_min + (q_temp_max - q_temp_min) / 2
        else:
            q_temp_max = q_temp_max - (q_temp_max - q_temp_min) / 2

    ans = (q_temp_min + q_temp_max) / 2
    return ans


def direction_assertions(direction: str, q_min: float, q_max: float):
    """Ensure correct values of q_min and q_max based on direction."""
    exist = 1
    if direction == "positive":
        if q_max < 1:
            exist = 0
        elif q_min < 1:
            q_min = 1
    else:
        if q_min > 1:
            exist = 0
        elif q_max > 1:
            q_max = 1

    return exist, q_min, q_max
