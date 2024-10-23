"""
This module contains code for MDSS algorithm.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import cpu_count

from functools import partial
import operator
from typing import Union

import numpy as np
import pandas as pd

from mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from mdss.ScoringFunctions.BerkJones import BerkJones
from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.ScoringFunctions.Poisson import Poisson
from mdss.ScoringFunctions.Gaussian import Gaussian

from mdss.generator import get_entire_subset, get_random_subset
from mdss.utils import check_inputs_len, reset_indexes
from mdss.contiguous_feature import get_contiguous_set_indices


@dataclass
class MDSSData:
    """
    Class holds data for MDSS module.
    """

    def __init__(
        self,
        coordinates: pd.DataFrame,
        outcomes: pd.Series,
        expectations: Union[pd.Series, pd.DataFrame],
        penalty: Union[float, None],
        num_iters: int,
        contiguous: dict,
        feature_penalty: float,
        verbose: bool,
        seed: int,
        num_of_subsets: int,
        mode: str,
        cpu: float,
        use_not_direction: bool,
        max_literals: int
    ) -> None:
        self.coordinates = coordinates
        self.outcomes = outcomes
        self.expectations = expectations
        self.penalty = penalty
        self.num_iters = num_iters
        self.contiguous = contiguous
        self.feature_penalty = feature_penalty
        self.verbose = verbose
        self.seed = seed
        self.num_of_subsets = num_of_subsets
        self.mode = mode
        self.cpu = cpu
        self.use_not_direction = use_not_direction
        self.max_literals = max_literals


class MDSS:
    """
    Multidimensional subset scanning (mdscan).

    Given a dataset `D` with outcomes `Y` and discretized features `X`.
    Also given `E` to be a set of expectations or 'normal' values for `Y`,
    and `F` to be an expectation-based scoring statistic that measures the
    amount of anomalousness between subgroup observations and their expectations.
    MDScan efficiently identifies the most anomalous subgroup; `S^*`.
    """

    def __init__(self, scoring_function: ScoringFunction):
        self.scoring_function = scoring_function

        # Stores scanning results during ascents
        self.best_scores = []
        self.best_subsets = []
        self.starting_subsets = []

        # For storing scanning results in nominal mode
        self._all_nominal = {}
        self.all_nominal_rankings = None

    def translate_subset(
        self,
        coordinates: pd.DataFrame,
        subset: dict,
        ):
        translated_subset = {}
        for key, value in subset.items():
            if isinstance(value, list):
                translated_subset[key] = value
            elif isinstance(value, set):
                all_categories = coordinates[key].unique()
                value = [i for i in all_categories if i not in value]
                translated_subset[key] = value
            else:
                assert False, "Should be list or set"
        return translated_subset

    def get_aggregates(
        self,
        coordinates: pd.DataFrame,
        outcomes: pd.Series,
        expectations: pd.Series,
        current_subset: dict,
        column_name: str,
        penalty: float,
        is_attr_contiguous: bool,
        use_not_direction: bool,
    ):
        """
        Conditioned on the current subsets of values for all other attributes,
        compute the summed outcome (observed_sum = sum_i y_i) and all expectations
        for each value of the current attribute.
        Also use additive linear-time subset scanning to compute the set of distinct thresholds
        for which different subsets of attribute values have positive scores. Note that the number
        of such thresholds will be linear rather than exponential in the arity of the attribute.

        :param coordinates: data frame containing having as columns the covariates/features
        :param expectations: data series containing the expected outcomes
        :param outcomes: data series containing the observed outcomes
        :param current_subset: current subset to compute aggregates
        :param column_name: attribute name to scan over
        :param penalty: penalty coefficient
        :param is_attr_contiguous: is the current attribute contiguous
        :return: dictionary of aggregates, sorted thresholds (roots),
                observed sum of the subset, array of observed outcomes.
        """
        # compute the subset of records matching the current subgroup along all other dimensions
        # temp_df includes the covariates x_i, outcome y_i, and expectationfor each matching record
        if current_subset:
            translated_subset = self.translate_subset(coordinates, current_subset)
            to_choose = (
                coordinates[translated_subset.keys()].isin(translated_subset).all(axis=1)
            )
            temp_df = pd.concat(
                [
                    coordinates.loc[to_choose],
                    outcomes[to_choose],
                    expectations[to_choose],
                ],
                axis=1,
            )
        else:
            temp_df = pd.concat([coordinates, outcomes, expectations], axis=1)

        # these wil be used to keep track of the aggregate values
        # and the distinct thresholds to be considered
        aggregates = {}
        thresholds = set()

        scoring_function = self.scoring_function

        # consider each distinct value of the given attribute (column_name)
        for name, group in temp_df.groupby(column_name):
            # Update the group when using the NOT direction
            if use_not_direction:
                group = temp_df.loc[[i for i in temp_df.index if i not in group.index]]

            # compute the sum of outcomes \sum_i y_i
            observed_sum = group.iloc[:, -2].sum()

            # all expectations
            if use_not_direction:
                expectations = group.iloc[:, -1]
            else:
                expectations = group.iloc[:, -1].values

            # compute q_min and q_max for the attribute value
            exist, q_mle, q_min, q_max = scoring_function.compute_qs(
                observed_sum,  np.array(expectations), penalty
            )

            # Add to aggregates, and add q_min and q_max to thresholds.
            # Note that thresholds is a set so duplicates will be removed automatically.
            if is_attr_contiguous:
                aggregates[name] = {
                    "observed_sum": observed_sum,
                    "expectations": expectations,
                }
            else:
                if exist:
                    aggregates[name] = {
                        "q_mle": q_mle,
                        "q_min": q_min,
                        "q_max": q_max,
                        "observed_sum": observed_sum,
                        "expectations": expectations,
                    }
                    thresholds.update([q_min, q_max])

        # We also keep track of the summed outcomes \sum_i y_i
        # and the expectations for the case where
        # all values of that attribute are considered
        # (regardless of whether they contribute positively to score).
        # This is necessary because of the way we compute the penalty term:
        # including all attribute values, equivalent
        # to ignoring the attribute, has the lowest penalty (of 0)
        # and thus we need to score that subset as well.
        
        all_observed_sum = temp_df.iloc[:, -2].sum()
        
        if use_not_direction:
            all_expectations = temp_df.iloc[:, -1]
        else:
            all_expectations = temp_df.iloc[:, -1].values

        val = [aggregates, sorted(thresholds), all_observed_sum, all_expectations]
        return val

    def choose_aggregates(
        self,
        aggregates: dict,
        thresholds: list,
        penalty: float,
        all_observed_sum: float,
        all_expectations: np.array,
        feature_penalty: float,
    ):
        """
        Having previously computed the aggregates and the distinct q thresholds
        to consider in the get_aggregates function,we are now ready to choose the best
        subset of attribute values for the given attribute.
        For each range defined by these thresholds,
        we will choose all of the positive contributions,
        compute the MLE value of q, and the corresponding score.
        We then pick the best q and score over all of the ranges considered.

        :param aggregates: dictionary of aggregates.
                        For each feature value, it has q_mle, q_min, q_max, observed_sum,
        and the expectations
        :param thresholds: sorted thresholds (roots)
        :param penalty: penalty coefficient
        :param all_observed_sum: sum of observed binary outcomes for all i
        :param all_expectations: data series containing all the expected outcomes
        :param feature_penalty (optional): extra penalty for the number of features in S*
        :return [best subset (of attribute values), best score]:
        """
        # initialize
        best_score = 0
        best_names = []

        scoring_function = self.scoring_function

        # for each threshold
        for i in range(len(thresholds) - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2
            observed_sum = 0.0
            expectations = []
            names = []

            # keep only the aggregates
            # which have a positive contribution to the score in that q range
            # we must keep track of the sum of outcome values
            # as well as all predicted outcomes
            for key, value in aggregates.items():
                if (value["q_min"] < threshold) & (value["q_max"] > threshold):
                    names.append(key)
                    observed_sum += value["observed_sum"]
                    expectations = expectations + value["expectations"].tolist()

            if len(expectations) == 0:
                continue

            # compute the MLE value of q,
            # making sure to only consider the desired direction (positive or negative)
            expectations = np.asarray(expectations)
            current_q_mle = scoring_function.qmle(observed_sum, expectations)

            # Compute the score for the given subset at the MLE value of q.
            # Notice that each included value gets a penalty, so the total penalty
            # is multiplied by the number of included values.
            penalty_ = (penalty * len(names))

            current_interval_score = scoring_function.score(
                observed_sum, expectations, penalty_, current_q_mle
            )

            # keep track of the best score, best q, and best subset of attribute values found so far
            if current_interval_score > best_score:
                best_score = current_interval_score
                best_names = names

        # Now we also have to consider the case of including all attribute values,
        # including those that never make positive contributions to the score.
        # Note that the penalty term is 0 in this case.  (We are neglecting penalties
        # from all other attributes, just considering the current attribute.)

        # compute the MLE value of q,
        # making sure to only consider the desired direction (positive or negative)
        current_q_mle = scoring_function.qmle(all_observed_sum, all_expectations)

        # Compute the score for the given subset at the MLE value of q.
        # Again, the penalty (for that attribute) is 0 when all attribute values are included.

        current_score = scoring_function.score(
            all_observed_sum, all_expectations, 0, current_q_mle
        )

        # Keep track of the best score, best q, and best subset of attribute values found.
        # Note that if the best subset contains all values of the given attribute,
        # we return an empty list for best_names.
        if current_score > best_score - feature_penalty:
            best_score = current_score
            best_names = []

        return [best_names, best_score]

    def choose_aggregates_not(
        self,
        aggregates: dict,
        thresholds: list,
        penalty: float,
        all_observed_sum: float,
        all_expectations: pd.Series,
        feature_penalty: float,
    ):
        """
        Having previously computed the aggregates and the distinct q thresholds
        to consider in the get_aggregates function,we are now ready to choose the best
        subset of attribute values for the given attribute.
        For each range defined by these thresholds,
        we will choose all of the positive contributions,
        compute the MLE value of q, and the corresponding score.
        We then pick the best q and score over all of the ranges considered.

        :param aggregates: dictionary of aggregates.
                        For each feature value, it has q_mle, q_min, q_max, observed_sum,
        and the expectations
        :param thresholds: sorted thresholds (roots)
        :param penalty: penalty coefficient
        :param all_observed_sum: sum of observed binary outcomes for all i
        :param all_expectations: data series containing all the expected outcomes
        :param feature_penalty (optional): extra penalty for the number of features in S*
        :return [best subset (of attribute values), best score]:
        """
        # initialize
        best_score = 0
        best_names = []

        scoring_function = self.scoring_function

        # for each threshold
        for i in range(len(thresholds) - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2
            observed_sum = all_observed_sum
            expectations = all_expectations
            names = set()

            # keep only the aggregates
            # which have a positive contribution to the score in that q range
            # we must keep track of the sum of outcome values
            # as well as all predicted outcomes
            for key, value in aggregates.items():
                if (value["q_min"] < threshold) & (value["q_max"] > threshold):
                    names.add(key)
                    observed_sum -= all_observed_sum - value["observed_sum"]
                    idx = [
                        i
                        for i in expectations.index
                        if i in value["expectations"].index
                    ]
                    expectations = expectations.loc[idx]

            if len(expectations) == 0:
                continue

            # compute the MLE value of q,
            # making sure to only consider the desired direction (positive or negative)

            expectations = np.asarray(expectations)
            current_q_mle = scoring_function.qmle(observed_sum, expectations)

            # Compute the score for the given subset at the MLE value of q.
            # Notice that each included value gets a penalty, so the total penalty
            # is multiplied by the number of included values.

            penalty_ = (penalty * len(names))

            current_interval_score = scoring_function.score(
                observed_sum, expectations, penalty_, current_q_mle
            )

            # print('current interval score, names - ', current_interval_score, names)
            # print()
            # keep track of the best score, best q, and best subset of attribute values found so far
            if current_interval_score > best_score:
                best_score = current_interval_score
                best_names = names

        # Now we also have to consider the case of including all attribute values,
        # including those that never make positive contributions to the score.
        # Note that the penalty term is 0 in this case.  (We are neglecting penalties
        # from all other attributes, just considering the current attribute.)

        # compute the MLE value of q,
        # making sure to only consider the desired direction (positive or negative)
        current_q_mle = scoring_function.qmle(all_observed_sum, all_expectations)

        # Compute the score for the given subset at the MLE value of q.
        # Again, the penalty (for that attribute) is 0 when all attribute values are included.

        current_score = scoring_function.score(
            all_observed_sum, all_expectations, 0, current_q_mle
        )

        # Keep track of the best score, best q, and best subset of attribute values found.
        # Note that if the best subset contains all values of the given attribute,
        # we return an empty list for best_names.
        if current_score > best_score - feature_penalty:
            best_score = current_score
            best_names = set()

        return [best_names, best_score]

    def choose_connected_aggregates(
        self,
        aggregates: dict,
        penalty: float,
        all_observed_sum: float,
        all_expectations: np.array,
        feature_penalty: float,
        contiguous_tuple: tuple
    ):
        """
        :param aggregates: dictionary of aggregates. For each feature value,
                                it has observed_sum, and the expectations
        :param penalty: penalty coefficient
        :param all_observed_sum: sum of observed binary outcomes for all i
        :param all_expectations: data series containing all the expected outcomes
        :param contiguous_tuple: tuple of order of the feature values,
                                and if missing or unknown value exists
        :param feature_penalty (optional): extra penalty for the number of features in S*
        :return [best subset (of attribute values), best score]:
        """
        best_names = []
        best_score = current_score = -1e10
        scoring_function = self.scoring_function

        contiguous_set_indices = get_contiguous_set_indices(contiguous_tuple[0])

        all_feature_values = contiguous_tuple[0]

        # we score the O(k^2) ranges of contiguous indices
        # for each contiguous range in the set of ranges
        for contiguous_subset in contiguous_set_indices:
            # no counts and no expectations
            observed_sum = 0.0
            expectations = []

            # for each bin in the range
            for feature_value_index in contiguous_subset:
                feature_value = all_feature_values[feature_value_index]

                if feature_value in aggregates.keys():
                    observed_sum += aggregates[feature_value]["observed_sum"]
                    expectations = (
                        expectations
                        + aggregates[feature_value]["expectations"].tolist()
                    )

            expectations_arr = np.array(expectations)
            current_q_mle = scoring_function.qmle(observed_sum, expectations_arr)

            # we only penalize the range irrespective of the number of bins once
            current_score = scoring_function.score(
                observed_sum=observed_sum,
                expectations=expectations_arr,
                penalty=penalty,
                q=current_q_mle,
            )

            if current_score > best_score:
                best_names = [all_feature_values[i] for i in contiguous_subset]
                best_score = current_score

        # the case where there is 'missing' data in this feature;
        if contiguous_tuple[1] is not None and contiguous_tuple[1] in aggregates.keys():
            for contiguous_subset in contiguous_set_indices:
                # take into consideration the counts and expectations of missing records
                observed_sum = aggregates[contiguous_tuple[1]]["observed_sum"]
                expectations = aggregates[contiguous_tuple[1]]["expectations"].tolist()

                # for each bin in the range
                for feature_value_index in contiguous_subset:
                    feature_value = all_feature_values[feature_value_index]

                    if feature_value in aggregates.keys():
                        observed_sum += aggregates[feature_value]["observed_sum"]
                        expectations = (
                            expectations
                            + aggregates[feature_value]["expectations"].tolist()
                        )

                expectations_arr = np.array(expectations)
                current_q_mle = scoring_function.qmle(observed_sum, expectations_arr)

                # We penalize once for the range and once for the missing bin
                current_score = scoring_function.score(
                    observed_sum=observed_sum,
                    expectations=expectations_arr,
                    penalty=2 * penalty,
                    q=current_q_mle,
                )

                if current_score > best_score:
                    best_names = [all_feature_values[i] for i in contiguous_subset] + [
                        contiguous_tuple[1]
                    ]
                    best_score = current_score

            # scanning over records that only have missing values
            observed_sum = aggregates[contiguous_tuple[1]]["observed_sum"]
            expectations = aggregates[contiguous_tuple[1]]["expectations"].tolist()

            expectations_arr = np.array(expectations)
            current_q_mle = scoring_function.qmle(observed_sum, expectations_arr)

            # we penalize once for the missing bin

            current_score = scoring_function.score(
                observed_sum=observed_sum,
                expectations=expectations_arr,
                penalty=penalty,
                q=current_q_mle,
            )

            if current_score > best_score:
                best_names = [contiguous_tuple[1]]
                best_score = current_score

        # cover the all case:
        # Again, the penalty (for that attribute) is 0 when all attribute values are included.

        current_q_mle = scoring_function.qmle(all_observed_sum, all_expectations)
        current_score = scoring_function.score(
            observed_sum=all_observed_sum,
            expectations=all_expectations,
            penalty=0,
            q=current_q_mle,
        )

        if current_score > best_score - feature_penalty:
            best_names = []
            best_score = current_score

        return [best_names, best_score]

    def score_current_subset(
        self,
        coordinates: pd.DataFrame,
        outcomes: pd.Series,
        expectations: pd.Series,
        current_subset: dict,
        penalty: float,
        feature_penalty: float,
        contiguous: dict,
    ):
        """
        Just scores the subset without performing ALTSS.
        We still need to determine the MLE value of q.

        :param coordinates: data frame containing having as columns the covariates/features
        :param outcomes: data series containing the observed outcomes
        :param expectations: data series containing the expected outcomes
        :param current_subset: current subset to be scored
        :param penalty: penalty coefficient
        :param contiguous (optional): contiguous features and thier order
        :param feature_penalty (optional): extra penalty for the number of features in S*
        :return: penalized score of subset
        """

        # compute the subset of records matching the current subgroup along all dimensions
        # temp_df includes the covariates x_i, outcome y_i, and expectationfor each matching record
        if current_subset:
            translated_subset = self.translate_subset(coordinates, current_subset)

            to_choose = (
                coordinates[translated_subset.keys()].isin(translated_subset).all(axis=1)
            )
            temp_df = pd.concat(
                [
                    coordinates.loc[to_choose],
                    outcomes[to_choose],
                    expectations[to_choose],
                ],
                axis=1,
            )
        else:
            temp_df = pd.concat([coordinates, outcomes, expectations], axis=1)

        scoring_function = self.scoring_function

        # we must keep track of the sum of outcome values as well as all predicted outcomes
        observed_sum = temp_df.iloc[:, -2].sum()
        expectations = temp_df.iloc[:, -1].values

        # compute the MLE value of q,
        # making sure to only consider the desired direction (positive or negative)
        current_q_mle = scoring_function.qmle(observed_sum, expectations)

        # total_penalty = penalty * sum of list lengths in current_subset
        # need to change to cater to fact that contiguous value count penalty once

        total_penalty = 0
        for key, values in current_subset.items():
            if key in list(contiguous.keys()):
                if len(values) == 1:
                    total_penalty += 1

                elif contiguous[key][1] in values:
                    total_penalty += 2

                else:
                    total_penalty += 1
            else:
                total_penalty += len(values)

        total_penalty *= penalty

        # Extra penalty for number of features are added
        extra_penalty = (len(current_subset.items()) * feature_penalty)
        total_penalty += extra_penalty

        # Compute and return the penalized score
        penalized_score = scoring_function.score(
            observed_sum, expectations, total_penalty, current_q_mle
        )

        return penalized_score

    def _do_checks(self, coordinates, outcomes, expectations, mode, cpu):
        """Validate data passed in by user.

        Raises:
            Exception: raises assertion errors
        """
        # Check cpu proportion
        assert 0 <= cpu <= 1, f"CPU proportion should be between 0 and 1, got {cpu}"
        # Check mode
        modes = set(["binary", "nominal", "ordinal", "continuous"])
        assert mode in modes, f"Expected one of {modes}, got {mode}."

        # Ensure that input pandas objects have same length and indexes
        assert check_inputs_len(
            coordinates, outcomes, expectations
        ), "Input Pandas objects do not have the same length"
        reset_indexes(coordinates, outcomes, expectations)

        unique_expectations = expectations.unique()

        if len(unique_expectations) == 1:
            autostrat_mode = True
        else:
            autostrat_mode = False

        if isinstance(self.scoring_function, Gaussian):
            assert mode == "continuous", f"Expected continuous, got {mode}."

        # Check that the appropriate scoring function is used
        if isinstance(self.scoring_function, BerkJones) and autostrat_mode is False:
            raise Exception(
                "BerkJones scorer supports scanning in autostrat mode only."
            )

        if isinstance(self.scoring_function, Bernoulli):
            modes = ["binary", "nominal"]
            assert mode in modes, f"Expected one of {modes} for Bernoulli,  got {mode}."

        if isinstance(self.scoring_function, Poisson):
            modes = ["binary", "ordinal"]
            assert mode in modes, f"Expected one of {modes} for Poisson,  got {mode}."

    def _scan_single_ascent_helper(
        self, data, current_subset, attribute_to_scan, is_attr_contiguous, use_not
    ):
        """
        The _scan_single_ascent_helper function is a helper function that is called by the scan_single_ascent method.
        It takes in a data object, current subset dictionary, attribute to scan (a string),
        and whether or not the attribute is contiguous (a boolean).
        It then calls get_aggregates and choose_aggregates to find best subset of attribute values.

        :param data: Get the coordinates, outcomes and expectations
        :param current_subset: Keep track of the current subset of features
        :param attribute_to_scan: Determine which attribute to scan
        :param is_attr_contiguous: Determine whether the attribute to scan is contiguous or not
        :param use_not: Determine whether the function should use the not_direction of the feature or not
        :return: A tuple of the following form:
        """

        # Only scan for NOT if feature values is great than two,
        # even when the NOT flag is set to true because for binary features,
        # A = 0 and A = Not 1 will give the same search space
        not_binary = data.coordinates[attribute_to_scan].nunique() > 2
        use_not = (use_not is True) and (not_binary is True)

        # call get_aggregates and choose_aggregates to find best subset of attribute values

        (
            aggregates,
            thresholds,
            all_observed_sum,
            all_expectations,
        ) = self.get_aggregates(
            coordinates=data.coordinates,
            outcomes=data.outcomes,
            expectations=data.expectations,
            current_subset=current_subset,
            column_name=attribute_to_scan,
            penalty=data.penalty,
            is_attr_contiguous=is_attr_contiguous,
            use_not_direction=use_not,
        )

        if is_attr_contiguous:
            temp_subset = current_subset.copy()

            temp_names, temp_score = self.choose_connected_aggregates(
                aggregates=aggregates,
                penalty=data.penalty,
                all_observed_sum=all_observed_sum,
                all_expectations=all_expectations,
                contiguous_tuple=data.contiguous[attribute_to_scan],
                feature_penalty=data.feature_penalty,
            )

            if temp_names:
                temp_subset[attribute_to_scan] = temp_names

            temp_score = self.score_current_subset(
                coordinates=data.coordinates,
                outcomes=data.outcomes,
                expectations=data.expectations,
                penalty=data.penalty,
                current_subset=temp_subset,
                contiguous=data.contiguous,
                feature_penalty=data.feature_penalty,
            )

        else:
            temp_subset = current_subset.copy()

            if use_not: # Only call choose_aggregates_not for non-contiguous values
                temp_names, temp_score = self.choose_aggregates_not(
                    aggregates=aggregates,
                    thresholds=thresholds,
                    penalty=data.penalty,
                    all_observed_sum=all_observed_sum,
                    all_expectations=all_expectations,
                    feature_penalty=data.feature_penalty,
                )

            else:
                temp_names, temp_score = self.choose_aggregates(
                    aggregates=aggregates,
                    thresholds=thresholds,
                    penalty=data.penalty,
                    all_observed_sum=all_observed_sum,
                    all_expectations=all_expectations,
                    feature_penalty=data.feature_penalty,
                )

            if temp_names:
                temp_subset[attribute_to_scan] = temp_names

            temp_score = self.score_current_subset(
                coordinates=data.coordinates,
                outcomes=data.outcomes,
                expectations=data.expectations,
                penalty=data.penalty,
                current_subset=temp_subset,
                contiguous=data.contiguous,
                feature_penalty=data.feature_penalty
            )
        return temp_subset, temp_score

    def _scan_single_ascent(self, data: MDSSData, current_subset: dict):
        """
        The _scan_single_ascent function is a helper function that is called by the scan_ascent method.
        It takes in a single MDSSData object and returns the best subset of attributes found during its search.
        The _scan_single_ascent function first initializes flags to 0,
        which indicates that we have not yet scanned all attributes for this data set.
        We then iterate until all flags are 1,
        indicating that we have scanned all attributes for this data set (and therefore cannot improve score any further).

        :param self: Reference the class instance
        :param data:MDSSData: Store the data passed to each of the methods
        :param current_subset:dict: Store the attribute values that have been chosen for a given subset
        :return: A tuple of the current_subset and current_score
        """
        # flags indicates that the method has optimized over subsets for a given attribute.
        # The iteration ends when it cannot further increase score by optimizing over
        # subsets of any attribute, i.e., when all flags are 1.
        flags = np.empty(len(data.coordinates.columns))
        flags.fill(0)

        # score the entire population
        current_score = self.score_current_subset(
            coordinates=data.coordinates,
            outcomes=data.outcomes,
            expectations=data.expectations,
            penalty=data.penalty,
            current_subset=current_subset.copy(),
            contiguous=data.contiguous,
            feature_penalty=data.feature_penalty,
        )

        while flags.sum() < len(data.coordinates.columns):

            # choose random attribute that we haven't scanned yet
            attribute_number_to_scan = np.random.choice(len(data.coordinates.columns))
            while flags[attribute_number_to_scan]:
                attribute_number_to_scan = np.random.choice(
                    len(data.coordinates.columns)
                )
            attribute_to_scan = data.coordinates.columns.values[
                attribute_number_to_scan
            ]

            # Storing these for edge cases in NOT search space
            feature_values_before_scanning = current_subset.get(attribute_to_scan, None)

            # Checking flag.sum() because temp_score only exist after one scan
            score_before_scanning = temp_score if flags.sum() > 0 else 0

            # clear current subset of attribute values for that subset
            if attribute_to_scan in current_subset:
                del current_subset[attribute_to_scan]

            is_attr_contiguous = attribute_to_scan in data.contiguous.keys()

            # Prime direction is always scanned
            temp_subset_prime, temp_score_prime = self._scan_single_ascent_helper(
                data,
                current_subset.copy(),
                attribute_to_scan,
                is_attr_contiguous,
                False,
            )

            if data.use_not_direction:
                temp_subset_not, temp_score_not = self._scan_single_ascent_helper(
                    data,
                    current_subset.copy(),
                    attribute_to_scan,
                    is_attr_contiguous,
                    True,
                )

                # This deals with the edge cases where LTSS property may not hold in NOT
                # TODO: Check this mathematically
                if temp_score_not < score_before_scanning:
                    temp_score_not = score_before_scanning

                    # If the feature exists in the subset before scanning,
                    # reassign the feature values before scanning since these values gave an higher score.
                    # If the feature did not exist in the subset before scanning,
                    # remove the feature values after scanning if scanning added the feature.

                    if feature_values_before_scanning:
                        temp_subset_not[
                            attribute_to_scan
                        ] = feature_values_before_scanning
                    else:
                        # Using pop  with None because scanning might not have returned the feature value
                        # and del temp_subset_not[attribute_to_scan] will throw a KeyError in that case.
                        temp_subset_not.pop(attribute_to_scan, None)
            else:
                temp_score_not = temp_score_prime - 1
                temp_subset_not = {}

            # If scores of both direction are equal,
            # take the subset of the direction with the smaller number of literals,
            # else, take the subset of the maximum score of the two
            if abs(temp_score_prime - temp_score_not) < 1e-6:
                num_literals_prime = sum([len(val) for _, val in temp_subset_prime.items()])
                num_literals_not = sum([len(val) for _, val in temp_subset_not.items()])

                temp_score, temp_subset = (
                    [temp_score_prime, temp_subset_prime]
                    if num_literals_prime <= num_literals_not
                    else [temp_score_not, temp_subset_not]
                )

            else:

                temp_score, temp_subset = (
                [temp_score_prime, temp_subset_prime]
                if  temp_score_prime >  temp_score_not
                else [temp_score_not, temp_subset_not]
                )

            # reset flags to 0 if we have improved score
            if temp_score > current_score + 1e-6:
                flags.fill(0)

            # sanity check to make sure score has not decreased
            # sanity check may not apply to Gaussian in penalized mode
            # TODO: to check Maths again for Gaussian
            if (not isinstance(self.scoring_function, Gaussian)) and (data.penalty > 0):
                assert (
                    temp_score >= current_score - 1e-6
                ), f"WARNING SCORE HAS DECREASED from {current_score:.6f} to {temp_score:.6f}"

            flags[attribute_number_to_scan] = 1
            current_subset = temp_subset
            current_score = temp_score

        return current_subset, current_score


    def _scan_ascents_single_core(
        self,
        data: MDSSData,
        starting_subset: dict = None,
    ):
        """
        Runs the scanning algorithm using a single-core

        :param data: MDSS_Data: Data for scanning
        :return: [best subset, best score]
        """
        if starting_subset is None:
            starting_subset = get_entire_subset()

        np.random.seed(data.seed)

        if isinstance(self.scoring_function, BerkJones):
            # Bin the continuous outcomes column for Berk Jones in continuous mode
            alpha = self.scoring_function.alpha
            direction = self.scoring_function.direction
            if data.mode == "continuous":
                quantile = data.outcomes.quantile(alpha)
                data.outcomes = (data.outcomes > quantile).apply(int)

            # Flip outcomes to scan in the negative direction for BerkJones
            # This is equivalent to switching the p-values
            if direction == "negative":
                data.outcomes = 1 - data.outcomes

        if isinstance(self.scoring_function, Gaussian):
            if self.scoring_function.mode == 'multiplicative':
                # Move entire distribution to the positive axis
                shift = np.abs(data.expectations.min()) + np.abs(data.outcomes.min())
                data.outcomes = data.outcomes + shift
                data.expectations = data.expectations + shift

        for key in data.contiguous.keys():
            assert (
                key in data.coordinates.columns
            ), "Contiguous key {key} missing in data."
            contiguous_values = data.contiguous[key]
            uniques = data.coordinates[key].unique().tolist()

            # Keep structure of contiguous graph after first scan
            for value in contiguous_values:
                if value not in uniques:
                    uniques.append(value)

            binslen = len(contiguous_values)
            uniquelen = len(uniques)

            assert binslen in [
                uniquelen - 1,
                uniquelen,
            ], f"""The attribute values {set(contiguous_values)} in the ordered list
                for contiguous feature {key} does not match 
                the attribute values {set(uniques)} in the data"""

            missing_bin_value = None
            if binslen == uniquelen - 1:
                missing_bin_values = set(uniques).difference(set(contiguous_values))
                assert (
                    len(missing_bin_values) == 1
                ), "More than one missing feature from contiguous features."
                missing_bin_value = list(missing_bin_values)[0]

            data.contiguous[key] = (data.contiguous[key], missing_bin_value)

        # initialize
        best_subset = {}
        best_score = -1e10

        if data.cpu == 0:
            self.starting_subsets.append(starting_subset)
            self.starting_subsets = self.starting_subsets + [
            get_random_subset(
                data.coordinates, np.random.rand(1).item(), 10, data.contiguous
            )
            for _ in range(data.num_iters - 1)
        ]

        for i in range(data.num_iters):
            # Note that we start with all values for the first iteration if single core ascent
            # and random values for succeeding iterations.
            current_subset = self.starting_subsets[i] if data.cpu == 0 else starting_subset

            current_subset, current_score = self._scan_single_ascent(
                data, current_subset
            )

            if data.cpu == 0:
                self.best_scores.append(current_score)
                self.best_subsets.append(current_subset)

            # print out results for current iteration
            if data.verbose:
                print(
                    f"""Subset found on iteration {i + 1} of {data.num_iters}
                    with score {current_score} :\n{current_subset}."""
                )

            # update best_score and best_subset if necessary
            if current_score > best_score:
                best_subset = current_subset.copy()
                best_score = current_score

                if data.verbose:
                    print(f"Best score is now {best_score}.\n")

            elif data.verbose:
                print(
                    f"Current score of {current_score} does not beat best score of {best_score}.\n"
                )
                
        return best_subset, best_score

    def _scan_ascents_in_parallel(self, data: MDSSData):
        """
        Run scan on multiple cores if they are available

        :param data: MDSS_Data: Data for scanning
        :param starting_subset: Starting subset for multiple ascents in single-core use.
        :return: [best subset, best score]
        """
        num_processes = int(cpu_count() * data.cpu)
        np.random.seed(data.seed)

        if num_processes > 1:
            self.starting_subsets.append(get_entire_subset())
            self.starting_subsets = self.starting_subsets + [
                get_random_subset(
                    data.coordinates, np.random.rand(1).item(), 10, data.contiguous
                )
                for _ in range(data.num_iters - 1)
            ]


            data.num_iters = 1
            scan = partial(self._scan_ascents_single_core, data)

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                results = executor.map(scan, self.starting_subsets)

            # collect the results
            results = list(results)

            # get the best score and sub-population
            best_subset, best_score = max(results, key=operator.itemgetter(1))

            self.best_subsets = self.best_subsets + [result[0] for result in results]
            self.best_scores = self.best_scores + [result[1] for result in results]

        else:
            data.cpu = 0
            # single thread
            best_subset, best_score = self._scan_ascents_single_core(data)

        return best_subset, best_score

    def _scan_k_subsets(self, data: MDSSData):
        """
        Returns top-k anomalous subsets in one direction.

        :param data: MDSS_Data: Data for scanning
        :return: [[best subset 1, best score 1], ... [best subset 2, best score 2]]
        """
        if data.cpu == 0:
            scan_func = self._scan_ascents_single_core
        else:
            scan_func = self._scan_ascents_in_parallel

        k_subsets_and_scores = []
        subset = {}

        for _ in range(data.num_of_subsets):
            if subset:
                to_choose = data.coordinates[subset.keys()].isin(subset).all(axis=1)
                data.coordinates = data.coordinates[~to_choose]
                data.outcomes = data.outcomes[~to_choose]
                data.expectations = data.expectations[~to_choose]

            if data.penalty is None:
                data.penalty = 1e-3
                subset, score = scan_func(data)

                num_of_literals = sum([len(val) for _, val in subset.items()])
                data.penalty = score/num_of_literals

                while num_of_literals > data.max_literals:
                    subset, score = scan_func(data)
                    num_of_literals = sum([len(val) for _, val in subset.items()])
                    data.penalty = score/num_of_literals
            
            else:
                subset, score = scan_func(data)

            k_subsets_and_scores.append([subset, score])

        return k_subsets_and_scores

    def _scan_in_nominal_mode(self, data: MDSSData):
        """
        Returns scanning results for nominal_mode using one-vs-all mode.
        Returns only the results for the most anomalous category.
        Stores additional data in self._all_nominal and self._all_nominal_rankings.

        :param data: MDSS_Data: Data for scanning
        :return: [[best subset 1, best score 1], ... [best subset 2, best score 2]]
        """
        unique_outs = set(sorted(data.outcomes.unique()))
        size_unique_outs = len(unique_outs)
        expectations_cols = set(sorted(data.expectations.columns))

        assert (
            size_unique_outs <= 100
        ), f"Nominal mode only support up to 100 labels, got {size_unique_outs}."

        assert (
            unique_outs == expectations_cols
        ), f"Expected {unique_outs} in expectation columns, got {expectations_cols}"

        self.all_nominal_rankings = dict(zip(unique_outs, [0] * size_unique_outs))

        for i, out in enumerate(unique_outs):
            if data.verbose:
                print(f"Scanning over outcome {i + 1} of {size_unique_outs}.")
            mapped_outcomes = data.outcomes.map({out: 1})
            mapped_outcomes.fillna(0, inplace=True)

            k_subsets_and_scores = self._scan_k_subsets(data)

            for _, score in k_subsets_and_scores:
                self.all_nominal_rankings[out] += score

            self._all_nominal[out] = k_subsets_and_scores

        max_key = max(self.all_nominal_rankings, key=self.all_nominal_rankings.get)
        return self._all_nominal[max_key]

    def _scan_in_diff_modes(self, data: MDSSData):
        """
        Returns top-k anomalous subsets in one direction for the different modes.

        :param data: MDSS_Data: Data for scanning
        :return: [[best subset 1, best score 1], ... [best subset 2, best score 2]]
        """
        if data.mode == "nominal":
            return self._scan_in_nominal_mode(data)

        return self._scan_k_subsets(data)

    def scan(
        self,
        coordinates: Union[pd.DataFrame, np.ndarray],
        outcomes: Union[pd.Series, np.ndarray],
        expectations: Union[pd.Series, pd.DataFrame, np.ndarray],
        penalty: Union[float, None],
        num_iters: int,
        max_literals: int = 5,
        use_not_direction: bool = False,
        contiguous: dict = None,
        feature_penalty: float = 0.0,
        verbose: bool = False,
        seed: int = 0,
        num_of_subsets: int = 1,
        mode: str = "binary",
        cpu: float = 0,
    ):
        """
        :param coordinates: data frame or numpy array containing having as columns the covariates/features
        :param outcomes: data series or numpy array containing the outcomes/observed outcomes
        :param expectations: data series or numpy array containing the expected outcomes.
                If mode == 'nominal', this is a dataframe with columns
                containing expectations for each nominal class.
        :param penalty: penalty coefficient
        :param num_iters: number of iteration
        :param max_literals: max number of literals to include in the returned subset
        :param use_not_direction (optional): flag to include the not search space
        :param contiguous (optional): contiguous features and their order
        :param feature_penalty (optional): extra penalty for the number of features in S*
        :param verbose: logging flag
        :param seed: numpy seed. Default equals 0
        :param num_of_subsets: number of anomalous subsets and scores to return
        :param mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
                In nominal mode, up to 100 categories are supported by default.
        :param cpu: between 0 and 1 the proportion of cpus available to use to scan.
                    Used to compute number of cores to run scan on in parallel.
                    Defaults to 0 for single-core scan.
        :return: [[best subset 1, best score 1], ... [best subset k, best score k]]
        """
        if isinstance(coordinates, np.ndarray):
            n_features = coordinates.shape[1]
            cols = range(n_features)
            coordinates = pd.DataFrame(coordinates, columns=cols)

        if isinstance(outcomes, np.ndarray):
            outcomes = pd.Series(outcomes)
        
        if isinstance(expectations, np.ndarray):
            if mode == 'nominal':
                assert False, """Nominal mode expects a dataframe with columns
                containing expectations for each nominal class."""
            expectations = pd.Series(expectations)

        self._do_checks(coordinates, outcomes, expectations, mode, cpu)

        if contiguous is None:
            contiguous = {}

        data = MDSSData(
            coordinates = coordinates,
            outcomes = outcomes,
            expectations = expectations,
            penalty = penalty,
            num_iters = num_iters,
            contiguous = contiguous,
            feature_penalty = feature_penalty,
            verbose = verbose,
            seed = seed,
            num_of_subsets = num_of_subsets,
            mode = mode,
            cpu = cpu,
            use_not_direction = use_not_direction,
            max_literals = max_literals
            )

        results = self._scan_in_diff_modes(data)

        if (
            len(results) == 1
        ):  # Return the subset and score separately if num_of_subsets = 1
            # Ensure backward compatibility with earlier notebooks.
            subset, score = results[0]
            return subset, score

        return results
