from numpy import cumsum
def split_array(vector, n_processors):
    n = len(vector)
    n_portions, rest = divmod(n, n_processors)
    counts = [0] + [n_portions + 1 if p<rest else n_portions for p in range(n_processors)]
    counts = cumsum(counts)
    start_end = zip(counts[:-1], counts[1:])
    slice_list = (slice(*sl) for sl in list(start_end))
    return [vector[sl] for sl in slice_list]
