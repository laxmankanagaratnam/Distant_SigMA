import numpy as np
from skopt.sampler import Lhs
from scipy.stats import qmc
from itertools import product


def generate_sfs(loc_to_file):
    # read in stds
    std_vals = np.genfromtxt(loc_to_file, usecols=(1, 2, 3), skip_header=0, max_rows=5)
    sfs_array = np.empty(shape=(5, 3))
    for h, row in enumerate(std_vals[:]):
        flipped_row = row[::-1]  # need to flip because I am using the inverses?
        sfs_array[h] = 1 / flipped_row

    return sfs_array


def random_sampling(sfs_file, size, seed=42):
    np.random.seed(seed)
    sfs_array = generate_sfs(sfs_file)
    ra_scaling, dec_scaling, plx_scaling, pmra_scaling, pmdec_scaling = sfs_array

    combinations = np.array(list(product(ra_scaling, dec_scaling, plx_scaling, pmra_scaling, pmdec_scaling)))

    sampled_rows = np.random.choice(combinations.shape[0], size=size, replace=False)
    # Use the sampled rows to extract the corresponding entries
    sampled_entries = combinations[sampled_rows]
    # Calculate the mean of each column
    column_means = np.mean(sampled_entries, axis=0)

    return sampled_entries, column_means


def lhs_minmax_center_sampling(sfs_file , size, seed=42):

    sfs_array = generate_sfs(sfs_file)

    lhs = Lhs(criterion="maximin", iterations=10000)
    samples = lhs.generate(sfs_array, size, seed)

    sample_array = np.array(samples).reshape(size, 5)
    column_means = np.mean(sample_array, axis=0)

    return sample_array, column_means


def lhc_lloyd(sfs_file, size):

    sfs_array = generate_sfs(sfs_file)

    l_bound = list(sfs_array[:, 0])
    h_bound = list(sfs_array[:, 2])

    # Sanity check
    # print("lower: ",l_bound)
    # print("upper: ", h_bound)

    sampler = qmc.LatinHypercube(d=len(l_bound), optimization="lloyd")
    samples = sampler.random(n=size)

    scaled_samples = qmc.scale(samples, l_bound, h_bound)
    column_means = np.mean(scaled_samples, axis=0)

    return scaled_samples, column_means


# if __name__ == "__main__":
#     sfs, column_means = lhc_lloyd(
#         "/Users/alena/PycharmProjects/SigMA_Orion/Data/Scale_factors/" + f"sfs_region_0.0.txt", 10)
#     print(sfs, column_means)