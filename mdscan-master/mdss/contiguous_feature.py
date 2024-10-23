import pandas as pd
import numpy as np


def get_contiguous_set_indices(feature_values: list):
    """
    This creates the set of possible contiguous features values
    give the entire ordered list of feature values.
    """

    contiguous_bins_index = []
    l = len(feature_values)

    for i in range(l):
        bin_list = [i]

        contiguous_bins_index.append(bin_list.copy())
        cycle_complement = []
        for j in range(i + 1, l):

            ## get bins without cycles
            bin_list.append(j)

            contiguous_bins_index.append(bin_list.copy())
            cycle_complement.append(j)

            ## get bins with cycles
            # if j != l - 1:
            #     cycle = set(range(l)).difference(cycle_complement)
            #     contiguous_bins_index.append(list(cycle))

    return contiguous_bins_index
