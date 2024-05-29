import numpy as np
import pandas as pd
import sys
import os
from itertools import product
from sklearn.metrics import normalized_mutual_info_score as nmi
from astropy.coordinates import ICRS, GalacticLSR
import astropy.units as u

from SigMA.SigMA import SigMA
from Loop_functions import setup_ICRS_ps, remove_field_stars, extract_signal_remove_spurious, extract_signal, \
    save_output_summary, consensus_function
from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility
from PlotlyResults import plot


# --------------------- Functions ---------------------
def noise_generator(N_signal: int, signal_to_noise: float, ranges: np.array):
    N_noise = int(N_signal / signal_to_noise)
    # print(N_noise)
    uniform_distribution = np.vstack([np.random.uniform(min_val, max_val, size=N_noise) for min_val, max_val in
                                      ranges]).T

    return uniform_distribution


def spherical2GalacticLSR(spherical_data):
    coord = ICRS(
        ra=spherical_data[0] * u.deg, dec=spherical_data[1] * u.deg, distance=spherical_data[2] * u.pc,
    )
    d = coord.transform_to(GalacticLSR())
    d.representation_type = 'cartesian'

    return [d.x.value, d.y.value, d.z.value]


def scale_factors(filepath: str, c_solution: int):
    if c_solution == 6:
        stds = np.genfromtxt(filepath, usecols=(1, 2, 3), skip_header=2, max_rows=5)
    elif c_solution == 4:
        stds = np.genfromtxt(filepath, usecols=(1, 2, 3), skip_header=10, max_rows=5)
    else:
        print("Only 4C and 6C solutions available at the moment")
        stds = None

    sfs = np.empty(shape=(5, 3))
    for h, row in enumerate(stds[:]):
        flipped_row = row[::-1]
        sfs[h] = 1 / flipped_row

    return sfs


# --------------------- Paths ---------------------
# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
output_path = my_utility.set_output_path(
    main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Sigma_Orion/Coding/Code_output/')
# print the parameter choices into log-textfile into the right folder
########
run = "SF-6C-CC-11"
########
if not os.path.exists(output_path + f"Run_{run}/"):
    os.makedirs(output_path + f"Run_{run}/")

output_path = output_path + f"Run_{run}/"

# read in data
labeled_clusters = pd.read_csv(
    "/Data/Region_0/Simulated_clusters_labeled_Region0_run6.csv")

# --------------------- Noise ---------------------
cols = ["ra", "dec", "plx", "pmra", "pmdec"]
noise_range = [(min(labeled_clusters[c]) - np.ptp(labeled_clusters[c]) * 0.3,
                max(labeled_clusters[c]) + np.ptp(labeled_clusters[c]) * 0.3)
               for c in cols]

# define SNR and 5D uniform distribution
#########
sn = 0.5
#########
unif = pd.DataFrame(data=noise_generator(labeled_clusters.shape[0], sn, noise_range),
                    columns=["ra", "dec", "plx", "pmra", "pmdec"]).assign(label=0)  # do not go smaller than 0.005

# Generate XYZ data for bg sources
XYZ_df = pd.DataFrame(np.column_stack(spherical2GalacticLSR([unif.ra, unif.dec, 1000 / unif.plx])),
                      columns=["X", "Y", "Z"])

# Concatenate the ICRS DataFrame and the XYZ DataFrame
unif_full = pd.concat([unif, XYZ_df], axis=1)

# merge the signal data with the noise data
merged_df = pd.merge(labeled_clusters, unif_full,
                     on=['ra', 'dec', 'plx', 'pmra', 'pmdec', 'label', 'X', 'Y', 'Z'], how='outer')

# --------------------- SigMA ---------------------
# define fixed SigMA parameters
step = 2
alpha = 0.05
beta = 0.99
knn_initcluster_graph = 35

############
knn = 20
bh = False
############
n_resampling = 0
scaling = None

feature_space = ['ra', 'dec', 'plx', 'pmra', 'pmdec']

# --------------------- Scale factors ---------------------
std_path = "/Data/Region_0/simulated_sfs.txt"
ra_scaling, dec_scaling, plx_scaling, pmra_scaling, pmdec_scaling = scale_factors(std_path, 6)

# create the 243 possible combinations
combinations = np.array(list(product(ra_scaling, dec_scaling, plx_scaling, pmra_scaling, pmdec_scaling)))

# sample 20 random combinations (with seed)
# Set a seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
# Sample 20 random rows from the array
sampled_rows = np.random.choice(combinations.shape[0], size=243, replace=False)

# Use the sampled rows to extract the corresponding entries
sampled_entries = combinations[sampled_rows]
# Calculate the mean of each column
column_means = np.mean(sampled_entries, axis=0)

# Print or use the column means
print(column_means)

# --------------------- Evaluate sampled gps ---------------------
# initialize SigMA for computational efficiency
setup_kwargs, df_focus = setup_ICRS_ps(df_fit=merged_df, sf_params=['ra', 'dec', 'plx'],
                                       sf_range=[ra_scaling, dec_scaling, plx_scaling], KNN_list=[knn], beta=beta,
                                       knn_initcluster_graph=knn_initcluster_graph, scaling=scaling)
sigma_kwargs = setup_kwargs["sigma_kwargs"]
scale_factor_list = setup_kwargs["scale_factor_list"]
clusterer = SigMA(
    data=df_focus, **sigma_kwargs)

# set mean of sampled list as SF
scale_factors = {'pos': {'features': ['ra', 'dec', 'plx'], 'factor': list(column_means[:3])},
                 'vel': {'features': ['pmra', 'pmdec'], 'factor': list(column_means[3:])}}
clusterer.set_scaling_factors(scale_factors)
print(clusterer.scale_factors)
# save X_mean
X_mean_sf = clusterer.X

label_matrix_rfs = np.empty(shape=(len(sampled_rows), len(df_focus)))
label_matrix_rsc = np.empty(shape=(len(sampled_rows), len(df_focus)))
label_matrix_simple = np.empty(shape=(len(sampled_rows), len(df_focus)))

# --------------------- Loop ---------------------
outer = np.empty(shape=(1, 6))
# outer_names = []
rho_sum = np.zeros(df_focus.shape[0], dtype=np.float32)

# Evaluate every grid point of the sample
for j, combo in enumerate(sampled_entries[:]):
    print(f"--- Gridpoint {j} ---")

    scale_factors = {'pos': {'features': ['ra', 'dec', 'plx'], 'factor': list(combo[:3])},
                     'vel': {'features': ['pmra', 'pmdec'], 'factor': list(combo[3:])}}
    #                 'vel': {'features': ['pmra', 'pmdec'], 'factor': [0.5,0.5]}}
    clusterer.set_scaling_factors(scale_factors)
    print(f"Performing clustering for scale factor {clusterer.scale_factors['pos']['factor']}...")

    # Fit
    clusterer.fit(alpha=alpha, knn=knn, bh_correction=bh)
    label_array = clusterer.labels_
    # density and X
    rho, X = clusterer.weights_, clusterer.X
    rho_sum += rho

    # a) remove field stars
    nb_rfs = remove_field_stars(label_array, rho, label_matrix_rfs, j)
    # b) remove spurious clusters
    nb_es, nb_rsc = extract_signal_remove_spurious(df_focus, label_array, rho, X, label_matrix_rsc, j)
    # c) extract signal
    nb_simple = extract_signal(label_array, clusterer, label_matrix_simple, j)

    labels_rsc = label_matrix_rsc[j, :].reshape(label_matrix_rsc.shape[1], )
    labels_rfs = label_matrix_rfs[j, :].reshape(label_matrix_rfs.shape[1], )
    labels_simple = label_matrix_simple[j, :].reshape(label_matrix_simple.shape[1], )

# Perform consensus clustering on the a) and b) arrays (automatically generates and saves a html-plot)
df_save = df_focus
label_lists = [label_matrix_rfs, label_matrix_rsc, label_matrix_simple]

# Perform consensus clustering on the c) and d) steps
labels_cc, n_cc = zip(
    *(consensus_function(jl, rho_sum, df_focus, f"Run_{run}_sn_{sn}_{name}_CC",
                         output_path, plotting=False) for jl, name in zip(label_lists, ["rfs", "rsc", "simple"])))
n_occ = list(n_cc)
labels_occ = list(labels_cc)

nmis = []
names = ["rfs", "rsc", "new"]
for i, entry in enumerate(labels_occ):
    print(f"{names[i]}: {np.unique(entry)} - nmi:", nmi(df_focus.label, entry))
    nmis.append(nmi(df_focus.label, entry))
    plot(entry, df_focus, f"Run_{run}_{names[i]}", output_path, icrs=True, return_fig=False)


save_output_summary(
    summary_str={"run": run, "knn": knn, "nmi_rfs": nmis[0], "nmi_rsc": nmis[1], "nmi_new": nmis[2],
                 "n_rfs": n_occ[0], "n_rsc": n_occ[1], "n_new": n_occ[2]},
    file=output_path + f"summary_run_{run}_sn_{sn}_CC.csv")

# save the labels in a csv file and plot the result
df_save["rsc"] = labels_occ[1]
df_save["rfs"] = labels_occ[0]
df_save["simple"] = labels_occ[2]
df_save.to_csv(output_path + f"Run_{run}_results_CC.csv")

outer[0,:3] = nmis
outer[0,3:] = [(len(np.unique(i)) - 1) for i in labels_occ]
summary_df = pd.DataFrame(data=outer, columns=["nmi_rfs", "nmi_rsc", "nmi_new", "n_rfs", "n_rsc", "n_new"])
summary_df.to_csv(output_path + f"{run}_sn_{sn}_knn_{knn}_bh_{bh}_summary.csv")

##########
# Output log-file
all_fixed = {"step": step, "alpha": alpha, "beta": beta, "knn_initcluster_graph": knn_initcluster_graph,
             "KNN": knn, "sfs_list": f"cc_{len(sampled_rows)}_seed_{seed_value}", "scaling": scaling, "bh_correction":
                 bh,
             names[np.argmax(nmis)]: np.max(nmis), "rsc": nmis[1]}

filename = output_path + f"Parameters_run_{run}.txt"
with open(filename, 'w') as file:
    for key, value in all_fixed.items():
        file.write(f"{key} = {value}\n")
###########