import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from misc import utilities as ut
from DistantSigMA.clustering_routine import run_clustering
from DistantSigMA.DistantSigMA.cluster_simulations import SimulateCluster
from DistantSigMA.DistantSigMA.PlotlyResults import plot
from DistantSigMA.DistantSigMA.sampling import lhc_lloyd
from DistantSigMA.DistantSigMA.noise_removal_functions import partial_clustering

from alex_workspace.ClusterHandler import ClusteringHandler
from alex_workspace.GraphCreator import GraphCreator
from alex_workspace.Tree import Custom_Tree
from alex_workspace.NxGraphAssistant import NxGraphAssistant
from alex_workspace.PlotHandler import PlotHandler


# -------------------- Setup ----------------------------
# plots
plot_figs = False

# Paths
sys.path.append('/Users/alena/PycharmProjects/Distant_SigMA')
script_name = ut.get_calling_script_name(__file__)
output_path = ut.set_output_path(script_name=script_name)

run = "test_combined_pipeline"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# -------------------- Data ----------------------------

# load the dataframe
df_load = pd.read_csv('../Data/Segments/Orion_labeled_segments_KNN_300_15-11-23.csv')

# load data for error sampling (already slimmed)
error_sampling_df = pd.read_csv("../Data/Gaia/Gaia_DR3_500pc_10percent.csv")


# =================== Clustering ========================

# ------------ A) Preliminary solution -------------------

# 1.) Parameters
dict_prelim = dict(alpha=0.01,
                   beta=0.99,
                   knn_initcluster_graph=35,
                   KNN_list=[15, 20, 25, 30],
                   sfs=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
                   scaling="robust",
                   bh_correction=True)

# 2.) Loop over each chunk
end_dfs = []
chunk_labels = df_load.region
for chunk in np.unique(chunk_labels)[:1]:

    print(f"-- Chunk: {chunk} --\n")
    print(f"PART A) Starting clustering ... \n")

    # extend output path
    result_path = output_path + f"Region_{int(chunk)}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # from now on only focus on this df
    df_chunk = df_load[df_load.region == chunk]

    # 3.) Preliminary solution from which the scale factors are determined
    df_prelim = run_clustering(region_label=chunk, df_input=df_chunk, sf_params="parallax_scaled",
                               parameter_dict=dict_prelim, mode="prelim", output_loc=result_path)

    # ------------ B) Simulate clusters -------------------
    print(f"PART B) Simulating clusters ... \n")

    # grab the clusters from the rsc solution
    df_clusters = df_prelim[df_prelim.rsc != -1]  # -1 == field stars

    # Add some small artificial clusters to increase scale factor sensitivity for those
    n_artificial = 1
    n_simulated_clusters = len(np.unique(df_clusters["rsc"])) + n_artificial

    # calculate center of all clusters
    cluster_features = ['X', 'Y', 'Z', 'v_a_lsr', "v_d_lsr"]
    kmeans = KMeans(n_clusters=1).fit(df_clusters[cluster_features])
    centers_real = kmeans.cluster_centers_[0]

    # initialize results arrays for the 5 cluster features
    stds = np.empty(shape=(5, n_simulated_clusters))
    e_convolved_dfs = []

    # Loop over the clusters
    for group in np.unique(df_clusters["rsc"])[:]:

        # define subset for length check
        subset = df_clusters[df_clusters["rsc"] == group]

        # Simulate the cluster from its covariance matrix and convolve it with Gaia errors
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
        if plot_figs:
            fig = sim.diff_histogram(e_convolved_cluster)
            plt.savefig(output_path + f"Run_{run}/"+f"Group_{sim.group_id}.pdf", dpi=300)

    # Create a master df of all groups
    convolved_df = pd.concat(e_convolved_dfs, ignore_index=True)

    if plot_figs:
        convolved_df.to_csv(output_path + f"Run_{run}/" + "Simulated_clusters_labeled_Region0_run10.csv")
        im_clusters = plot(convolved_df["label"], convolved_df, f"SIM_{run}", icrs=True, output_pathname=result_path)

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

        outer_fig.savefig(output_path + f"Run_{run}/"+"outer_distributions.png", dpi=300)

    # save scaling factors in Data directory
    directory = "/Users/alena/PycharmProjects/SigMA_Orion/Data/Scale_factors"
    filename = f"sfs_region_{chunk}.txt"

    # Open the file in write mode ('w')
    with open(f"{directory}/{filename}", 'w') as file:
        for i, label in enumerate(["ra", "dec", "parallax", "pmra", "pmdec"]):
            print(f"{label}:", np.min(stds[i, :]), np.mean(stds[i, :]), np.max(stds[i, :]), file=file)

    # ------------ C) Cluster with new SF -------------------
    print(f"PART C) Re-evaluating clustering ... \n")

    # determine the number of SF to draw using the lhc_lloyd sampling of the parameter space
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

    # Generate grouped solutions
    grouped_solutions, occ_solutions, rho_sum = partial_clustering(region_label=chunk, df_input=df_chunk,
                                                                   sf_params=["ra", "dec", "parallax", "pmra", "pmdec"],
                                                                   parameter_dict=dict_final, mode="final",
                                                                   output_loc=result_path,
                                                                   column_means=scale_factor_means)
    print(grouped_solutions.shape)

    # save the grouped solution labels to the dataframe holding the observations
    df_final = df_chunk.copy()
    for col in range(grouped_solutions.shape[0]):
        df_final.loc[:, f"cluster_label_group_{col}"] = grouped_solutions[col, :]
    df_final.to_csv(result_path+f"Region_{chunk}_sf_{num_sf}_grouped_solutions.csv")
    end_dfs.append(df_final)

    # ------------ D) Graph/Tree application -------------------







