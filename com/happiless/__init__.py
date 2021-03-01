import numpy as np

small_counts = np.random.randint(0, 100, 20)

np.floor_divide(small_counts, 10)

large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495, 91897, 44, 28, 7971, 926, 122, 22222]

np.floor(np.log10(large_counts))