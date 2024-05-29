import os
import sys
import pandas as pd
import numpy as np

from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility
from DistantSigMA.DistantSigMA.sampling import lhc_lloyd
from DistantSigMA.DistantSigMA.cluster_simulations import SimulateCluster
from DistantSigMA.DistantSigMA.PlotlyResults import plot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from DistantSigMA.DistantSigMA.noise_removal_functions import setup_ICRS_ps, extract_signal_remove_spurious, \
    remove_field_stars, extract_signal, \
    partial_clustering, save_output_summary, consensus_function
from SigMA.SigMA import SigMA


def run_clustering(region_label, df_input, sf_params, parameter_dict: dict, mode: str, output_loc: str,
                   column_means=None):
    # most important variables
    KNNs = parameter_dict["KNN_list"]
    # setup kwargs
    setup_kwargs, df_focus = setup_ICRS_ps(df_fit=df_input, sf_params=sf_params, sf_range=parameter_dict["sfs"],
                                           KNN_list=KNNs, beta=parameter_dict["beta"],
                                           knn_initcluster_graph=parameter_dict["knn_initcluster_graph"],
                                           scaling=parameter_dict["scaling"], means=column_means)
    sigma_kwargs = setup_kwargs["sigma_kwargs"]
    scale_factor_list = setup_kwargs["scale_factor_list"]

    # ---------------------------------------------------------

    # initialize SigMA with sf_mean
    clusterer = SigMA(data=df_focus, **sigma_kwargs)
    # save X_mean
    X_mean_sf = clusterer.X
    # initialize array for density values (collect the rho_sums)
    rhosum_list = []

    # Initialize array for the outer cc (occ) results (remove field stars / rfs, remove spurious clusters / rsc)
    results_rfs = np.empty(shape=(len(KNNs), len(df_focus)))
    results_rsc = np.empty(shape=(len(KNNs), len(df_focus)))
    results_simple = np.empty(shape=(len(KNNs), len(df_focus)))

    # Outer loop: KNN
    for kid, knn in enumerate(KNNs):

        print(f"-- Current run with KNN = {knn} -- \n")

        label_matrix_rfs = np.empty(shape=(len(scale_factor_list), len(df_focus)))
        label_matrix_rsc = np.empty(shape=(len(scale_factor_list), len(df_focus)))
        label_matrix_simple = np.empty(shape=(len(scale_factor_list), len(df_focus)))

        # initialize density-sum over all scaling factors
        rho_sum = np.zeros(df_focus.shape[0], dtype=np.float32)

        # ---------------------------------------------------------
        df_labels = pd.DataFrame()
        # Inner loop: Scale factors
        for sf_id, sf in enumerate(scale_factor_list):
            # Set current scale factor
            if mode == "prelim":
                scale_factors = {'pos': {'features': ['parallax_scaled'], 'factor': sf}}
                clusterer.set_scaling_factors(scale_factors)
                print(f"Performing clustering for scale factor {clusterer.scale_factors['pos']['factor']}...")
            elif mode == "final":
                scale_factors = {'pos': {'features': ['ra', 'dec', 'parallax'], 'factor': list(sf[:3])},
                                 'vel': {'features': ['pmra', 'pmdec'], 'factor': list(sf[3:])}}
                clusterer.set_scaling_factors(scale_factors)
                print(f"Performing clustering for scale factor p{clusterer.scale_factors['pos']['factor']}"
                      f"{clusterer.scale_factors['vel']['factor']}...")

            # Fit
            clusterer.fit(alpha=parameter_dict["alpha"], knn=knn, bh_correction=parameter_dict["bh_correction"])
            label_array = clusterer.labels_

            # density and X
            rho, X = clusterer.weights_, clusterer.X
            rho_sum += rho

            # a) remove field stars
            nb_rfs = remove_field_stars(label_array, rho, label_matrix_rfs, sf_id)
            # b) remove spurious clusters
            nb_es, nb_rsc = extract_signal_remove_spurious(df_focus, label_array, rho, X, label_matrix_rsc, sf_id)
            # c) do new method
            nb_simple = extract_signal(label_array, clusterer, label_matrix_simple, sf_id)
            # Write the output to the hyperparameter file:

            save_output_summary(
                summary_str={"knn": knn, "sf": str(sf), "n_rfs": nb_rfs, "n_rsc": nb_rsc, "n_simple": nb_simple},
                file=output_loc + f"{mode}_ICC_{knn}_summary.csv")

            df_labels[f"#_{sf_id}"] = label_matrix_rsc[sf_id, :]

        if mode == "final":
            # save output files
            df_labels.to_csv(output_loc + f"{mode}_ICC_{knn}_labels.csv")
        # append the density sum to the list over all KNN
        rhosum_list.append(rho_sum)

        # Perform consensus clustering on the a) and b) arrays (automatically generates and saves a html-plot)
        labels_icc_rfs, n_icc_rfs = consensus_function(label_matrix_rfs, rho_sum, df_focus,
                                                       f"{mode}_Region_{int(region_label)}_rfs_KNN_{knn}_ICC",
                                                       output_loc,
                                                       plotting=True)
        labels_icc_rsc, n_icc_rsc = consensus_function(label_matrix_rsc, rho_sum, df_focus,
                                                       f"{mode}_Region_{int(region_label)}_rsc_KNN_{knn}_ICC",
                                                       output_loc,
                                                       plotting=True)

        labels_icc_simple, n_icc_simple = consensus_function(label_matrix_simple, rho_sum, df_focus,
                                                             f"{mode}_Region_{int(region_label)}_simple_KNN_{knn}_ICC",
                                                             output_loc,
                                                             plotting=True)
        results_rfs[kid, :] = labels_icc_rfs

        results_rsc[kid, :] = labels_icc_rsc
        results_simple[kid, :] = labels_icc_simple

        print(f":: Finished run for KNN={knn}! \n. Found {n_icc_rfs} / {n_icc_rsc} / {n_icc_simple} final clusters.")

    knn_mid = int(len(KNNs) / 2 - 1)
    df_save = df_focus.copy()
    label_lists = [results_rfs, results_rsc, results_simple]

    # Perform consensus clustering on the c) and d) steps
    labels_occ, n_occ = zip(
        *(consensus_function(jl, rhosum_list[knn_mid], df_focus, f"{mode}_Region_{int(region_label)}_{name}_OCC",
                             output_loc) for jl, name in zip(label_lists, ["rfs", "rsc", "simple"])))
    n_occ = list(n_occ)
    labels_occ = list(labels_occ)

    # save the labels in a csv file and plot the result
    df_save["rsc"] = labels_occ[1]
    df_save["rfs"] = labels_occ[0]
    df_save["simple"] = labels_occ[2]
    df_save.to_csv(result_path + f"{mode}_Region_{int(region_label)}_results_CC.csv")

    save_output_summary(summary_str={"knn": "occ", "n_rfs": n_occ[0], "n_rsc": n_occ[1],
                                     "n_rfs_cleaned": n_occ[2]},
                        file=output_loc + f"{mode}_Region_{int(region_label)}_outer_summary.csv")

    # Output log-file
    filename = result_path + f"Region_{region_label}_{mode}_parameters.txt"
    with open(filename, 'w') as file:
        for key, value in parameter_dict.items():
            file.write(f"{key} = {value}\n")

    return df_save


# Paths
# ---------------------------------------------------------
# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
script_name = my_utility.get_calling_script_name(__file__)
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/', script_name=script_name)

run = "Orion_200_samples"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
# 2. Data
# ---------------------------------------------------------
# load the dataframe
df_load = pd.read_csv('../Data/Segments//Orion_labeled_segments_KNN_300_15-11-23.csv')

# load data for error sampling (already slimmed)
error_sampling_df = pd.read_csv("../Data/Gaia/Gaia_DR3_500pc_10percent.csv")

# =========================================================
# Part A) Preliminary solution
# =========================================================

# 1.) Parameters
# ---------------------------------------------------------

dict_prelim = dict(alpha=0.01,
                   beta=0.99,
                   knn_initcluster_graph=35,
                   KNN_list=[15, 20, 25, 30],
                   sfs=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
                   scaling="robust",
                   bh_correction=True)

# 4.) Loop over each chunk
# ---------------------------------------------------------
# initialize SigMA with sf_mean

end_dfs = []

chunk_labels = df_load.region
for chunk in np.unique(chunk_labels)[:]:

    print(f"-- Chunk: {chunk} --\n")
    print(f"PART A) Starting clustering ... \n")
    result_path = output_path + f"Region_{int(chunk)}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # from now on only focus on this df
    df_chunk = df_load[df_load.region == chunk]

    df_prelim = run_clustering(region_label=chunk, df_input=df_chunk, sf_params="parallax_scaled",
                               parameter_dict=dict_prelim, mode="prelim", output_loc=result_path)

    # =========================================================
    # Part B) Simulate clusters
    # =========================================================

    print(f"PART B) Simulating clusters ... \n")
    # features to calculated distribution for
    cluster_features = ['X', 'Y', 'Z', 'v_a_lsr', "v_d_lsr"]

    # grab the clusters from the rsc solution
    df_clusters = df_prelim[df_prelim.rsc != -1]

    # define number of simulated clusters
    n_artificial = 1
    n_simulated_clusters = len(np.unique(df_clusters["rsc"])) + n_artificial

    # calculate center of all clusters
    kmeans = KMeans(n_clusters=1).fit(df_clusters[cluster_features])
    centers_real = kmeans.cluster_centers_[0]

    stds = np.empty(shape=(5, n_simulated_clusters))

    e_convolved_dfs = []
    for group in np.unique(df_clusters["rsc"])[:]:

        # define subset for length check
        subset = df_clusters[df_clusters["rsc"] == group]
        print(group, subset.shape)

        sim = SimulateCluster(region_data=df_clusters, group_id=group, clustering_features=cluster_features)
        e_convolved_cluster = sim.error_convolve(sampling_data=error_sampling_df)
        sim_df = pd.DataFrame(data=sim.e_convolved_points,
                              columns=["ra", "dec", "parallax", "pmra", "pmdec", "X", "Y", "Z"]) \
            .assign(label=int(group))
        e_convolved_dfs.append(sim_df)

        # std cols
        std_columns = ["ra", "dec", "parallax", "pmra", "pmdec"]
        stds[:, group] = sim_df[std_columns].std().values

        # use smallest subset to add a tiny cluster in the center
        if len(subset) == df_clusters.groupby("rsc").size().min():
            sim.add_mini_cluster(n_members=max(dict_prelim['KNN_list']), center_coords=centers_real,
                                 pos_cov_frac=0.5)
            e_conv_tiny = sim.error_convolve(sampling_data=error_sampling_df)
            tiny_df = pd.DataFrame(data=sim.e_convolved_points,
                                   columns=["ra", "dec", "parallax", "pmra", "pmdec", "X", "Y", "Z"]).assign(
                label=n_simulated_clusters)
            e_convolved_dfs.append(tiny_df)

            # also add this to the std cols
            stds[:, n_simulated_clusters - 1] = tiny_df[std_columns].std().values

        # resampled data histograms
        fig = sim.diff_histogram(e_convolved_cluster)
        # plt.savefig(output_path + f"Run_{run}/"+f"Group_{sim.group_id}.pdf", dpi=300)

    convolved_df = pd.concat(e_convolved_dfs, ignore_index=True)
    # convolved_df.to_csv(output_path + f"Run_{run}/" + "Simulated_clusters_labeled_Region0_run10.csv")

    # plot the simulated clusters
    sim_clusters = plot(convolved_df["label"], convolved_df, f"SIM_{run}", icrs=True, output_pathname=result_path)

    # outer histogram of the group stds
    outer_fig, ax = plt.subplots(2, 3, figsize=(7, 4))
    ax = ax.ravel()

    try:
        for i, label in enumerate(["ra", "dec", "parallax", "pmra", "pmdec"]):
            data_column = stds[i, :]
            num_bins_data = sim.Knuths_rule(data_column)
            ax[i].hist(data_column, bins=len(data_column), facecolor="green", edgecolor='black')
            ax[i].set_title(f"{label} ({round(min(data_column), 3)}, {round(max(data_column), 3)})")

        plt.suptitle(f"Outer hist")
        plt.tight_layout()
        plt.close()
    except ValueError:
        pass
    # outer_fig.savefig(output_path + f"Run_{run}/"+"outer_distributions.png", dpi=300)

    # save scaling factors in Data directory
    directory = "/Users/alena/PycharmProjects/SigMA_Orion/Data/Scale_factors"
    filename = f"sfs_region_{chunk}.txt"

    # Open the file in write mode ('w')
    with open(f"{directory}/{filename}", 'w') as file:
        for i, label in enumerate(["ra", "dec", "parallax", "pmra", "pmdec"]):
            print(f"{label}:", np.min(stds[i, :]), np.mean(stds[i, :]), np.max(stds[i, :]), file=file)

    # =========================================================
    # Part C) Employ the new scale factors
    # =========================================================

    print(f"PART C) Re-evaluating clustering ... \n")

    num_sf = 200

    sfs, means = lhc_lloyd('../Data/Scale_factors/' + f'sfs_region_{chunk}.txt', num_sf)

    scale_factor_means = {'pos': {'features': ['ra', 'dec', 'parallax'], 'factor': list(means[:3])},
                          'vel': {'features': ['pmra', 'pmdec'], 'factor': list(means[3:])}}

    dict_final = dict(alpha=0.01,
                      beta=0.99,
                      knn_initcluster_graph=35,
                      KNN_list=[30],
                      sfs=sfs,
                      scaling=None,
                      bh_correction=False)

    grouped_solutions, occ_solutions, rho_sum = partial_clustering(region_label=chunk, df_input=df_chunk,
                                                                   sf_params=["ra", "dec", "parallax", "pmra", "pmdec"],
                                                                   parameter_dict=dict_final, mode="final",
                                                                   output_loc=result_path,
                                                                   column_means=scale_factor_means)

    # labels_occ, n_occ = consensus_function(grouped_label_matrix, rho_sum, df_focus, f"{mode}_SimSol_KNN_{knn}_OCC",
    #                                       output_loc, plotting=True)

    print(grouped_solutions.shape)

    df_final = df_chunk.copy()
    for col in range(grouped_solutions.shape[0]):
        df_final.loc[:, f"cluster_label_group_{col}"] = grouped_solutions[col, :]

    df_final.to_csv(result_path+f"Region_{chunk}_sf_{num_sf}_grouped_solutions.csv")

    end_dfs.append(df_final)

'''
# Concatenate datasets and adjust labels
concatenated_dfs = []
label_counter = 0

for df in end_dfs:
    df_with_new_labels = df.copy()
    mask = df['cluster_label'] != -1
    df_with_new_labels.loc[mask, 'cluster_label'] += label_counter
    concatenated_dfs.append(df_with_new_labels)
    label_counter += df['cluster_label'].max() + 1

# Concatenate all dataframes together
region_label_df = pd.concat(concatenated_dfs, axis=0, ignore_index=True)

print(region_label_df, np.unique(region_label_df["cluster_label"]))

region_label_df.to_csv(output_path + "Large_df.csv")
'''
print("--------- Routine executed successfully ---------")
