import pandas as pd
import numpy as np


def get_entire_subset():
    """
    Returns the entire subset, which is an empty dictionary
    :return: empty dictionary
    """
    return {}


def get_random_subset(
    coordinates: pd.DataFrame, prob: float, min_elements: int = 0, contiguous: dict = {}
):
    """
    Returns a random subset
    :param coordinates: data frame containing having as columns the features
    :param prob: probability to select a value of a feature
    :param min_elements: minimum number of elements to be included in the randomly generated sub-population
    :return: dictionary representing a random sub-population
    """

    subset_random_values = {}
    shuffled_column_names = np.random.permutation(coordinates.columns.values)

    # consider each column once, in random order
    for column_name in shuffled_column_names:
        # get unique values of the current column
        temp = coordinates[column_name].unique()

        # if the feature is a contiguous feature
        if column_name in contiguous.keys():
            temp = contiguous[column_name][0]
            value_probs = np.random.random(len(temp))

            # each value is randomly assigned probabilities and
            # we set the starting feature value to be the value with the higest probability
            # but also check that it is higher than the probability of selecting any feature value

            if np.max(value_probs) > prob:
                start = np.argmax(value_probs)
                indices = [start]
                prob_tuple = (
                    prob,
                    1 - prob,
                )  # not sure if this should be prob or prob/2

                # set a probability mask to determine when to move up or down
                down_mask = value_probs < prob_tuple[0]
                up_mask = value_probs > prob_tuple[1]

                for down_bool, up_bool in zip(down_mask, up_mask):

                    down = down_bool and indices[-1] > 0
                    up = up_bool and indices[-1] < len(temp) - 1

                    indices.append(indices[-1] - down + up)

                subset_random_values[column_name] = [temp[i] for i in indices]

            else:
                subset_random_values[column_name] = []
        else:
            # include each attribute value with probability = prob
            mask_values = np.random.rand(len(temp)) < prob
            if mask_values.sum() < len(temp):
                # set values for the current column
                subset_random_values[column_name] = temp[mask_values].tolist()

        # compute the remaining records
        if subset_random_values:
            mask_subset = (
                coordinates[subset_random_values.keys()]
                .isin(subset_random_values)
                .all(axis=1)
            )
            remaining_records = len(coordinates.loc[mask_subset])

            # only filter on this attribute if at least min_elements records would be kept
            if remaining_records < min_elements:
                del subset_random_values[column_name]

    return subset_random_values
