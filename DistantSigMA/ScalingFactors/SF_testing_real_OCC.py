import numpy as np
import pandas as pd
import sys
import os
from itertools import product

from SigMA.SigMA import SigMA
from Loop_functions import setup_ICRS_ps, remove_field_stars, extract_signal_remove_spurious, extract_signal, \
    save_output_summary, consensus_function
from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility

# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
output_path = my_utility.set_output_path(
    main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Sigma_Orion/Coding/Code_output/')

region = 0.0
run = 2

df_load = pd.read_csv("/Data/Orion_labeled_segments_KNN_300_15-11-23.csv")
df_region = df_load[df_load.region == region]
result_path = output_path + f"Run_{run}/Region_{int(region)}/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# define fixed SigMA parameters
step = 2
alpha = 0.05
beta = 0.99
knn_initcluster_graph = 35
KNN_list = np.linspace(30, 15, 4, dtype=int)
bh = False
n_resampling = 0
scaling = None

feature_space = ['ra', 'dec', 'plx', 'pmra', 'pmdec']

std_path = "/Data/Region_0/simulated_sfs.txt"
ra_scaling, dec_scaling, plx_scaling, pmra_scaling, pmdec_scaling = scale_factors(std_path, 6)
pmdec_new = pmdec_scaling[1:]
pmra_new = pmra_scaling[:2]
# create the 243 possible combinations
combinations = np.array(list(product(ra_scaling, dec_scaling, plx_scaling, pmra_new, pmdec_new)))

seed_value = 42
np.random.seed(seed_value)

sampled_rows = np.random.choice(combinations.shape[0], size=108, replace=False)

# Use the sampled rows to extract the corresponding entries
sampled_entries = combinations[sampled_rows]
# Calculate the mean of each column
column_means = np.mean(sampled_entries, axis=0)

all_fixed = {"step": step, "alpha": alpha, "beta": beta, "knn_initcluster_graph": knn_initcluster_graph,
             "KNN_list": KNN_list, "sfs_list": sampled_entries, "scaling": scaling}

df_region.rename(columns={"parallax": "plx"}, inplace=True)

# ---------------------------------------------------------
# output path extension

if not os.path.exists(output_path + f"Run_{run}/"):
    os.makedirs(output_path + f"Run_{run}/")
filename = output_path + f"Run_{run}/parameters_run_{run}.txt"
with open(filename, 'w') as file:
    for key, value in all_fixed.items():
        file.write(f"{key} = {value}\n")

# Create a new dictionary excluding unwanted keys
fixed_params = {key: value for key, value in all_fixed.items() if key not in ["step", "alpha", "CMD_plots"]}

# ---------------------------------------------------------
# CLUSTERING
# initialize SigMA for computational efficiency
setup_kwargs, df_focus = setup_ICRS_ps(df_fit=df_region, sf_params=['ra', 'dec', 'plx'],
                                       sf_range=[ra_scaling, dec_scaling, plx_scaling], KNN_list=KNN_list, beta=beta,
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

# initialize SigMA with sf_mean
clusterer = SigMA(data=df_focus, **sigma_kwargs)
# save X_mean
X_mean_sf = clusterer.X
# initialize array for density values (collect the rho_sums)
rhosum_list = []

# Initialize array for the outer cc (occ) results (remove field stars / rfs, remove spurious clusters / rsc)
results_rfs = np.empty(shape=(len(KNN_list), len(df_focus)))
results_rsc = np.empty(shape=(len(KNN_list), len(df_focus)))
results_simple = np.empty(shape=(len(KNN_list), len(df_focus)))

# Outer loop: KNN
for kid, knn in enumerate(KNN_list):

    # print(f"-- Current run with KNN = {knn} -- \n")

    label_matrix_rfs = np.empty(shape=(len(scale_factor_list), len(df_focus)))
    label_matrix_rsc = np.empty(shape=(len(scale_factor_list), len(df_focus)))
    label_matrix_simple = np.empty(shape=(len(scale_factor_list), len(df_focus)))

    # initialize density-sum over all scaling factors
    rho_sum = np.zeros(df_focus.shape[0], dtype=np.float32)

    # ---------------------------------------------------------
    # Evaluate every grid point of the sample
    for j, combo in enumerate(sampled_entries[:]):
        print(f"--- Gridpoint {j} ---")

        scale_factors = {'pos': {'features': ['ra', 'dec', 'plx'], 'factor': list(combo[:3])},
                         'vel': {'features': ['pmra', 'pmdec'], 'factor': list(combo[3:])}}
        #                 'vel': {'features': ['pmra', 'pmdec'], 'factor': [0.5,0.5]}}
        clusterer.set_scaling_factors(scale_factors)
        print(f"Performing clustering for scale factor {clusterer.scale_factors['pos']['factor']}"
              f"{clusterer.scale_factors['vel']['factor']}...")
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
        # c) do new method
        nb_simple = extract_signal(label_array, clusterer, label_matrix_simple, j)

        # Write the output to the hyperparameter file:
        save_output_summary(
            summary_str={"knn": knn, "sf": str(combo), "n_rfs": nb_rfs, "n_rsc": nb_rsc, "n_simple": nb_simple},
            file=result_path + f"Inner_{knn}_summary.csv")

        # append the density sum to the list over all KNN
        rhosum_list.append(rho_sum)
        label_lists = [label_matrix_rfs, label_matrix_rsc, label_matrix_simple]
        labels_icc, n_icc = zip(
            *(consensus_function(jl, rho_sum, df_focus, f"Run_{run}_real_{name}_CC",
                                 output_path, plotting=False) for jl, name in
              zip(label_lists, ["rfs", "rsc", "simple"])))
        n_cc = list(n_icc)
        labels_cc = list(labels_icc)

        results_rfs[kid, :] = labels_cc[0]
        results_rsc[kid, :] = labels_cc[1]
        results_simple[kid, :] = labels_cc[2]

        print(f":: Finished run for KNN={knn}! \n. Found {n_cc[0]} / {n_cc[1]} / {n_cc[2]} final clusters.")

    knn_mid = int(len(KNN_list) / 2 - 1)
    df_save = df_focus
    label_lists = [results_rfs, results_rsc, results_simple]

    # Perform consensus clustering on the c) and d) steps
    labels_occ, n_occ = zip(
        *(consensus_function(jl, rhosum_list[knn_mid], df_focus, f"Step{step}_C_{int(region)}_{name}_OCC",
                             result_path) for jl, name in zip(label_lists, ["rfs", "rsc", "simple"])))
    n_occ = list(n_occ)
    labels_occ = list(labels_occ)

    save_output_summary(summary_str={"knn": "occ", "n_rfs": n_occ[0], "n_rsc": n_occ[1],
                                     "n_rfs_cleaned": n_occ[2]},
                        file=result_path + f"Region_{int(region)}_outer_summary.csv")

    # save the labels in a csv file and plot the result
    df_save["rsc"] = labels_occ[1]
    df_save["rfs"] = labels_occ[0]
    df_save["simple"] = labels_occ[2]
    df_save.to_csv(result_path + f"Step{step}_C_{int(region)}_results_CC.csv")

    # loop_dict = {"coord_sys": modes[step], "alpha": alpha, "fixed_params": fixed_params,
    #             "output_path": result_path, "plotting": False}

    # results = cluster_solutions(step=step, num=region, df_fit=df_focus, knn_list=KNN_list,
    #                            output_path=result_path, loop_dict=loop_dict)

print("--------- Routine executed successfully ---------")
