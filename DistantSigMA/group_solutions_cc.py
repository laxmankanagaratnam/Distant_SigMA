import os
import sys

import pandas as pd
import numpy as np

from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility
from DistantSigMA.DistantSigMA.sampling import lhc_lloyd
from DistantSigMA.DistantSigMA.noise_removal_functions import setup_ICRS_ps, extract_signal_remove_spurious, \
    plot_dendrogram, get_similar_solutions, cluster_solutions, remove_field_stars, extract_signal, \
    save_output_summary, consensus_function
from SigMA.SigMA import SigMA
from DistantSigMA.PlotlyResults import plot

from sklearn.metrics import normalized_mutual_info_score as nmi

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

end_dfs = []

chunk_labels = df_load.region
for chunk in np.unique(chunk_labels)[:]:

    print(f"-- Chunk: {chunk} --\n")
    result_path = output_path + f"Region_{int(chunk)}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # from now on only focus on this df
    df_chunk = df_load[df_load.region == chunk]

    sfs, means = lhc_lloyd('../Data/Scale_factors/' + f'sfs_region_{chunk}.txt', 200)

    scale_factor_means = {'pos': {'features': ['ra', 'dec', 'parallax'], 'factor': list(means[:3])},
                          'vel': {'features': ['pmra', 'pmdec'], 'factor': list(means[3:])}}

    dict_final = dict(alpha=0.01,
                      beta=0.99,
                      knn_initcluster_graph=35,
                      KNN_list=[30],
                      sfs=sfs,
                      scaling=None,
                      bh_correction=False)

    # most important variables
    KNNs = dict_final["KNN_list"]
    # setup kwargs
    setup_kwargs, df_focus = setup_ICRS_ps(df_fit=df_chunk, sf_params=["ra", "dec", "parallax", "pmra", "pmdec"],
                                           sf_range=dict_final["sfs"],
                                           KNN_list=KNNs, beta=dict_final["beta"],
                                           knn_initcluster_graph=dict_final["knn_initcluster_graph"],
                                           scaling=dict_final["scaling"], means=scale_factor_means)
    sigma_kwargs = setup_kwargs["sigma_kwargs"]
    scale_factor_list = setup_kwargs["scale_factor_list"]

    # ---------------------------------------------------------
    # initialize SigMA with sf_mean
    clusterer = SigMA(data=df_focus, **sigma_kwargs)

    # JUST ONE KNN INITIALLY
    knn = KNNs[0]

    print(f"-- Current run with KNN = {knn} -- \n")

    label_matrix_rfs = np.empty(shape=(len(scale_factor_list), len(df_focus)))
    label_matrix_rsc = np.empty(shape=(len(scale_factor_list), len(df_focus)))
    label_matrix_simple = np.empty(shape=(len(scale_factor_list), len(df_focus)))

    # initialize density-sum over all scaling factors
    rho_sum = np.zeros(df_focus.shape[0], dtype=np.float32)

    # ---------------------------------------------------------
    solutions = []

    # Inner loop: Scale factors
    for sf_id, sf in enumerate(scale_factor_list):
        scale_factors = {'pos': {'features': ['ra', 'dec', 'parallax'], 'factor': list(sf[:3])},
                         'vel': {'features': ['pmra', 'pmdec'], 'factor': list(sf[3:])}}
        clusterer.set_scaling_factors(scale_factors)
        print(f"Performing clustering for scale factor {sf_id}...")

        # Fit
        clusterer.fit(alpha=dict_final["alpha"], knn=knn, bh_correction=dict_final["bh_correction"])
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
            file=output_path + f"CC_gs_ICC_{knn}_summary.csv")

        solutions.append({'labels': label_matrix_rsc[sf_id, :]})

    all_labels = [sol['labels'] for sol in solutions]

    threshold = 0.8  # threshold for similarity score
    penalty = 0.1  # penalty for different number of clusters
    score_func = nmi  # could also use fms, but seems to be less performant
    # Hierarchical agglomerative clustering
    clusterer_hier = cluster_solutions(
        all_labels,
        score_func=score_func, score_threshold=threshold, penalty=penalty, linkage='complete'
    )

    d = plot_dendrogram(clusterer_hier, truncate_mode=None, color_threshold=1 - threshold)

    d.savefig(result_path+f"Dendrogram_chunk_{int(chunk)}.png", dpi=200)

    # remove groups with only one entry already here
    all_solutions = [sol_group for sol_group in
                     get_similar_solutions(clusterer_hier, all_labels, min_solutions_per_cluster=1) if
                     len(sol_group) > 1]

    grouped_label_matrix = np.empty(shape=(len(all_solutions), len(all_solutions[0][0][1])))

    for h, sol_group in enumerate(all_solutions):
        print(h, len(sol_group), "\n")

        label_matrix = np.empty(shape=(len(sol_group), len(sol_group[0][1])))
        for k, entry in enumerate(sol_group):
            label_matrix[k, :] = entry[1]

        # Perform consensus clustering on the c) and d) steps
        labels_cc, n_cc = consensus_function(label_matrix, rho_sum, df_focus,
                                             f"CC_gs_SimSol_{int(h)}_KNN_{knn}_CC",
                                             output_path,
                                             plotting=True)

        grouped_label_matrix[h, :] = labels_cc

    # save output
    grouped_label_matrix_int = grouped_label_matrix.astype(int)
    np.savetxt(result_path + f'grouped_solutions_chunk_{int(chunk)}.csv', grouped_label_matrix, fmt='%d', delimiter=",")
    np.savetxt(result_path + f'Density_chunk_{int(chunk)}.csv', rho_sum, delimiter=",")

    labels_occ, n_occ = consensus_function(grouped_label_matrix, rho_sum, df_focus, f"CC_gs_SimSol_KNN_{knn}_OCC",
                                           output_path, plotting=True)

    df_final = df_chunk.copy()
    df_final.loc[:, "cluster_label"] = labels_occ

    end_dfs.append(df_final)

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

region_label_df.to_csv(output_path + "Large_df_Orion-200.csv")

all_clusters = plot(region_label_df["cluster_label"], region_label_df, f"all_clusters", output_pathname=output_path)


print("--------- Routine executed successfully ---------")
