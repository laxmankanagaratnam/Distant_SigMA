import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score as nmi

from SigMA.SigMA import SigMA
from ConcensusClustering.consensus import ClusterConsensus

from DistantSigMA.DistantSigMA.setup_and_scaling import setup_ICRS_ps, save_output_summary
from DistantSigMA.DistantSigMA.noise_removal import remove_field_stars, extract_signal, \
    extract_signal_remove_spurious
from DistantSigMA.DistantSigMA.PlotlyResults import plot


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
    df_save.to_csv(output_loc + f"{mode}_Region_{int(region_label)}_results_CC.csv")

    save_output_summary(summary_str={"knn": "occ", "n_rfs": n_occ[0], "n_rsc": n_occ[1],
                                     "n_rfs_cleaned": n_occ[2]},
                        file=output_loc + f"{mode}_Region_{int(region_label)}_outer_summary.csv")

    # Output log-file
    filename = output_loc + f"Region_{region_label}_{mode}_parameters.txt"
    with open(filename, 'w') as file:
        for key, value in parameter_dict.items():
            file.write(f"{key} = {value}\n")

    return df_save


def consensus_function(label_matrix: np.array, density_sum: np.array, df_fit: pd.DataFrame, file: str = None,
                       path: str = None, plotting: bool = True, return_cc: bool = False):
    """
    Function that takes the different labels created in a loop over either KNNs or scaling factors and makes a consensus
    solution.

    :param label_matrix: matrix holding the different results of the loop
    :param density_sum: the sum of the 1D density calculated in each step of the loop
    :param df_fit: input dataframe
    :param file: filename for the plot
    :param path: output path of the plot
    :param plotting: bool
    :return: consensus-labels and number of clusters found in the consensus solution
    """
    cc = ClusterConsensus(*label_matrix)
    labels_cc = cc.fit(density=density_sum, min_cluster_size=15)
    labels_cc_clean = LabelEncoder().fit_transform(labels_cc) - 1
    if plotting:
        plot(labels=labels_cc_clean, df=df_fit, filename=file, output_pathname=path)
    nb_consensus = np.unique(labels_cc_clean[labels_cc_clean > -1]).size

    if return_cc:
        return cc, labels_cc_clean, nb_consensus
    else:
        return labels_cc_clean, nb_consensus


def score_similarity(all_labels, score_func, penalty=0.05):
    nmi_scores = np.zeros((len(all_labels), len(all_labels)))
    for i, labels_i in enumerate(all_labels):
        for j, labels_j in enumerate(all_labels):
            diff_size = abs(np.unique(labels_i).size - np.unique(labels_j).size)
            penalized_score = score_func(labels_i, labels_j) - penalty * diff_size
            nmi_scores[i, j] = max(penalized_score, 0)
    return nmi_scores


def cluster_solutions(all_labels, score_func=nmi, score_threshold=0.8, penalty=0.05, **kwargs):
    nmi_scores = score_similarity(all_labels, score_func, penalty=penalty)
    # Cluster with agglomerative clustering
    linkage = kwargs.pop('linkage', 'complete')
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - score_threshold, linkage=linkage, metric='precomputed', **kwargs
    ).fit(1 - nmi_scores)
    return clusterer


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    fig = plt.figure()
    dendrogram(linkage_matrix, **kwargs)
    return fig


def get_similar_solutions(clusterer, all_labels, min_solutions_per_cluster=2):
    cluster_idx = np.arange(len(clusterer.labels_))
    # Remove solutions with less than min_solutions_per_cluster
    unique_nb, counts = np.unique(clusterer.labels_, return_counts=True)
    clusters2investigate = unique_nb[counts >= min_solutions_per_cluster]

    for cluster_nb in clusters2investigate:
        ids = cluster_idx[clusterer.labels_ == cluster_nb]
        yield [(i, all_labels[i]) for i in ids]


def partial_clustering(df_input, sf_params, parameter_dict: dict, mode: str, output_loc: str,
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

    rhosum_list = []

    # Initialize array for the outer cc (occ) results (remove field stars / rfs, remove spurious clusters / rsc)
    results_rsc = np.empty(shape=(len(KNNs), len(df_focus)))

    solutions = []

    # Outer loop: KNN
    for kid, knn in enumerate(KNNs):

        # JUST ONE KNN INITIALLY
        #knn = KNNs[0]

        print(f"-- Current run with KNN = {knn} -- \n")

        label_matrix_rfs = np.empty(shape=(len(scale_factor_list), len(df_focus)))
        label_matrix_rsc = np.empty(shape=(len(scale_factor_list), len(df_focus)))
        label_matrix_simple = np.empty(shape=(len(scale_factor_list), len(df_focus)))

        # initialize density-sum over all scaling factors
        rho_sum = np.zeros(df_focus.shape[0], dtype=np.float32)

        # ---------------------------------------------------------

        # Inner loop: Scale factors
        for sf_id, sf in enumerate(scale_factor_list):
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

    plot_dendrogram(clusterer_hier, truncate_mode=None, color_threshold=1 - threshold)

    all_solutions = [sol_group for sol_group in
                     get_similar_solutions(clusterer_hier, all_labels, min_solutions_per_cluster=2)]

    grouped_label_matrix = np.empty(shape=(len(all_solutions), len(all_solutions[0][0][1])))

    for h, sol_group in enumerate(get_similar_solutions(clusterer_hier, all_labels, min_solutions_per_cluster=2)):
        # print(h, sol_group, "\n -----------------------------------")
        label_matrix = np.empty(shape=(len(sol_group), len(sol_group[0][1])))
        for k, entry in enumerate(sol_group):
            label_matrix[k, :] = entry[1]

        # Perform consensus clustering on the c) and d) steps
        labels_cc, n_cc = consensus_function(label_matrix, rho_sum, df_focus,
                                             f"{mode}_solutionGroup_{int(h)}_KNN_{knn}_CC",
                                             output_loc,
                                             plotting=True)

        grouped_label_matrix[h, :] = labels_cc

    # cc, labels_occ, n_occ = consensus_function(grouped_label_matrix, rho_sum, df_focus, f"{mode}_SimSol_KNN_{knn}_OCC",
    #                                       output_loc, plotting=True, return_cc=True)

    return grouped_label_matrix.astype(int)
