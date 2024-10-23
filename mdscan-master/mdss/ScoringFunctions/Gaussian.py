"""
Gaussian scoring function.
"""
import numpy as np
from mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from mdss.ScoringFunctions import optim


class Gaussian(ScoringFunction):
    """
    Gaussian score function. May be appropriate to use when the outcome of
    interest is assumed to be normally distributed.
    """

    def __init__(self, **kwargs):
        """
        kwargs must contain
        'direction (str)' - direction of the severity;
        could be higher than expected outcomes ('positive')
        or lower than expected ('negative')
        """
        self.var = kwargs.get('var')
        self.mode = kwargs.get('mode') if kwargs.get('mode') is not None else 'multiplicative'
        super(Gaussian, self).__init__(**kwargs)

    def score(
        self, observed_sum: float, expectations: np.array, penalty: float, q: float
    ):
        """
        Computes gaussian bias score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: bias score for the current value of q
        """
        eps = 1e-6
        assumed_var = self.var + eps
        expected_sum = expectations.sum()

        #if self.mode == 'multiplicative':
        c_term = observed_sum * expected_sum / assumed_var * (q - 1)
        b_term = expected_sum**2 * (1 - q**2) / (2 * assumed_var)
        #elif self.mode == 'additive':
        #    c_term = len(expectations) / assumed_var * q
        #    b_term = len(expectations) * q**2 / (2 * assumed_var)

        if c_term > b_term and self.direction == "positive":
            ans = c_term + b_term
        elif b_term > c_term and self.direction == "negative":
            ans = c_term + b_term
        else:
            ans = 0

        # elif self.mode == 'additive':
        #     if (self.direction == 'positive' and q > eps) or (self.direction == 'negative' and q < -eps):
        #         ans = len(expectations) * q**2 / (2 * assumed_var)
        #     else:
        #         ans = 0

        ans -= penalty
        # print('ans = ', ans)
        return ans

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).
        """
        eps = 1e-6
        expected_sum = expectations.sum() + eps
        observed_sum += eps

        if len(expectations) == 0:
            return 0

        if self.mode == 'multiplicative':            
            ans = observed_sum / expected_sum
        elif self.mode == 'additive':
            ans = (observed_sum - expected_sum)/len(expectations)
        
        return ans

    def compute_qs(self, observed_sum: float, expectations: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty coefficient
        """

        direction = self.direction

        q_mle = self.qmle(observed_sum, expectations)
        q_mle_score = self.score(observed_sum, expectations, penalty, q_mle)

        if q_mle_score > 0:
            exist = 1
            q_min = optim.bisection_q_min(
                self, observed_sum, expectations, penalty, q_mle, temp_min=-1e6
            )
            q_max = optim.bisection_q_max(
                self, observed_sum, expectations, penalty, q_mle, temp_max=1e6
            )
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        # # only consider the desired direction, positive or negative
        if exist:
            exist, q_min, q_max = optim.direction_assertions(direction, q_min, q_max)

        ans = [exist, q_mle, q_min, q_max]
        return ans
