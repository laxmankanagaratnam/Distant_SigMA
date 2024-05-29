import numpy as np
import pandas as pd
import sys
import os
from itertools import product
from SigMA.SigMA import SigMA
from Loop_functions import parameter_scaler, remove_field_stars, extract_signal_remove_spurious
from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.preprocessing import LabelEncoder
from PlotlyResults import plot


def noise_generator(N_signal: int, signal_to_noise: float, ranges: np.array):
    N_noise = int(N_signal / signal_to_noise)
    # print(N_noise)
    uniform_distribution = np.vstack([np.random.uniform(min_val, max_val, size=N_noise) for min_val, max_val in
                                      ranges]).T

    return uniform_distribution


# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/')

# read in data
labeled_clusters = pd.read_csv(
    "/Data/Region_0/Simulated_clusters_labeled_Region0.csv")

# read in region 0 as reference for the uniform distribution
region_0_all = pd.read_csv(
    "/Data/Region_0/Region_0_topcat_cleaned.csv")

# get limits for the uniform dist from region 0
noise_range = np.array([[min(region_0_all.ra), max(region_0_all.ra)],
                        [min(region_0_all.dec), max(region_0_all.dec)],
                        [min(region_0_all.parallax), max(region_0_all.parallax)],
                        [min(region_0_all.pmra), max(region_0_all.pmra)],
                        [min(region_0_all.pmdec), max(region_0_all.pmdec)]])

# define SNR and 5D uniform distribution
sn = 0.5
unif = pd.DataFrame(data=noise_generator(labeled_clusters.shape[0], sn, noise_range),
                    columns=["ra", "dec", "plx", "pmra", "pmdec"]).assign(label=0)  # do not go smaller than 0.005

# merge the signal data with the noise data
merged_df = pd.merge(labeled_clusters, unif, on=['ra', 'dec', 'plx', 'pmra', 'pmdec', 'label'], how='outer')

# define fixed SigMA parameters
step = 2
alpha = 0.01
beta = 0.99
knn_initcluster_graph = 35
knn = 20
robust_scaling = True
bh = False

# Setup phase space
n_resampling = 0

# normalize the data
df_scaled = merged_df.copy()
cols_to_scale = ['ra', 'dec', 'plx', 'pmra', 'pmdec']
scaled_cols = ['ra_scaled', 'dec_scaled', 'parallax_scaled', 'pmra_scaled', 'pmdec_scaled']
scaled_data = [parameter_scaler(df_scaled[col], robust_scaling) for col in cols_to_scale]
for col_id, col in enumerate(scaled_cols):
    df_scaled[col] = scaled_data[col_id]

# define scaling factors
ra_scaling = np.array([0.5823655787564496, 0.7707334636626149, 0.9865094976918362])
dec_scaling = np.array([0.6190326290424706, 0.7772915325254782, 1.005768259832928])
plx_scaling = np.array([0.14799044388147478, 0.15525138540355507, 0.1617409374077745])

# create the 27 possible combinations
combinations = np.array(list(product(ra_scaling, dec_scaling, plx_scaling)))

# print the parameter choices into log-textfile into the right folder
########
run = 2
########
if not os.path.exists(output_path + f"Run_{run}/"):
    os.makedirs(output_path + f"Run_{run}/")

for j, combo in enumerate(combinations[:]):
    print(f"--- Gridpoint {j} ---")

    # define columns and values of the scaling factors
    scale_factors = {'pos': {'features': ['ra_scaled', 'dec_scaled', 'parallax_scaled'], 'factor': list(combo)}}

    # SigMA kwargs
    sigma_kwargs = dict(cluster_features=scaled_cols, scale_factors=scale_factors, nb_resampling=n_resampling,
                        max_knn_density=knn + 1, beta=beta, knn_initcluster_graph=knn_initcluster_graph)

    # initialize SigMA
    clusterer = SigMA(data=df_scaled, **sigma_kwargs)
    label_array = clusterer.fit(alpha=alpha, knn=knn, bh_correction=bh)
    # density and X
    rho, X = clusterer.weights_, clusterer.X

    # old method
    label_matrix_rfs = np.empty(shape=(1, len(df_scaled)))
    label_matrix_rsc = np.empty(shape=(1, len(df_scaled)))
    # a) remove field stars
    nb_rfs = remove_field_stars(label_array, rho, label_matrix_rfs, 0)
    # b) remove spurious clusters
    nb_es, nb_rsc = extract_signal_remove_spurious(df_scaled, label_array, rho, X, label_matrix_rsc, 0)

    # new method
    labels_with_noise = extract_signal(label_array, clusterer)
    ln = LabelEncoder().fit_transform(labels_with_noise) - 1

    labels_rsc = label_matrix_rsc.reshape(label_matrix_rsc.shape[1], )
    labels_rfs = label_matrix_rfs.reshape(label_matrix_rfs.shape[1], )

    nmis = []
    names = ["new", "rsc", "rfs"]
    label_arrs = [ln, labels_rsc, labels_rfs]
    for i, entry in enumerate(label_arrs):
        print(f"{names[i]}: {np.unique(entry)} - nmi:", nmi(df_scaled.label, entry))
        nmis.append(nmi(df_scaled.label, entry))

    df_save = df_scaled
    df_save["rsc"] = label_matrix_rsc.T
    df_save["rfs"] = label_matrix_rfs.T
    df_save["new"] = labels_with_noise.T

    df_save.to_csv(output_path + f"Run_{run}/Step{step}_gridpoint_{j}_results.csv")

    ##########
    # Output log-file

    all_fixed = {"step": step, "alpha": alpha, "beta": beta, "knn_initcluster_graph": knn_initcluster_graph,
                 "KNN": knn, "sfs_list": combo, "robust_scaling": robust_scaling, "bh_correction": bh,
                 names[np.argmax(nmis)]: np.max(nmis)}

    filename = output_path + f"Run_{run}/parameters_gp_{j}.txt"
    with open(filename, 'w') as file:
        for key, value in all_fixed.items():
            file.write(f"{key} = {value}\n")

    plot(labels=label_arrs[np.argmax(nmis)], df=df_scaled, filename=f"Run_{run}_gp_{j}_best_nmi",
         output_pathname=output_path + f"Run_{run}/", icrs=True)

    ###########
