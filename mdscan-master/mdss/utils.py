def check_inputs_len(*args):
    """
    Ensures all the inputs to the scan method has the same length
    """
    lengths = []
    for arg in args:
        lengths.append(len(arg))

    return len(set(lengths)) == 1

def reset_indexes(*args):
    """
    Ensure all the input pandas objects have the same indexing
    """
    for arg in args:
        arg.reset_index(drop = True, inplace = True)

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def sort_subset_dict(subset):
    for key in subset:
        if key != "column_name":
            try:
                sorted_list = sorted(subset[key], key=lambda x: (len(x), x))
            except:
                sorted_list = sorted(subset[key])
            subset[key] = tuple(sorted_list)
    return hashabledict(subset)