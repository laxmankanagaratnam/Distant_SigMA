from NoiseRemoval.RemoveNoiseTransformed import remove_noise_quick_n_dirty, remove_noise_simple
from NoiseRemoval.ClusterSelection import nearest_neighbor_distribution, remove_outlier_clusters
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import LabelEncoder
from ConcensusClustering.consensus import ClusterConsensus
from DistantSigMA.DistantSigMA.PlotlyResults import plot
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.cluster.hierarchy import dendrogram

from DistantSigMA.DistantSigMA.clustering_setup import setup_ICRS_ps, save_output_summary

from SigMA.SigMA import SigMA

import matplotlib.pyplot as plt


def extract_signal_remove_spurious(df_fit: pd.DataFrame, labels: np.array, density: np.array, X_matrix: np.array,
                                   output_matrix: np.array, array_index: int):
    """
    Older function that I built by merging the old extract_signal and remove_spurious_clusters functions from the
    jupyter-notebook I started out with.

    :param df_fit: Input dataframe that was provided to the SigMA clusterer instance
    :param labels: Label array
    :param density: clusterer.weights attribute
    :param X_matrix: clusterer.X attribute
    :param output_matrix: matrix or 1,N array to write the new labels to. Matrix style was coded for loops over
    multiple sf or KNN
    :param array_index: row index of the output matrix specifying the actual position in the loop
    :return: The numbers of found clusters after a) removing field stars and b) removing spurious clusters. Labels are
    written to the output matrix.
    """

    # very similar to extract_signal, except for the use of a different noise removal function
    labels_with_noise = -np.ones(df_fit.shape[0], dtype=int)
    data_idx = np.arange(df_fit.shape[0])
    for ii, u_cl in enumerate(np.unique(labels[labels > -1])):
        cluster_bool_array, is_good_cluster = remove_noise_quick_n_dirty(density, labels == u_cl)
        if is_good_cluster:
            idx_cluster = data_idx[labels == u_cl][cluster_bool_array]
            labels_with_noise[idx_cluster] = ii

    nb_extracted_signal = np.unique(labels_with_noise[labels_with_noise > -1]).size

    # Compute nearest neighbors distances
    nn_data = nearest_neighbor_distribution(X_matrix)
    # Transform labels to start from 0 - (N-1)
    labels_traf = LabelEncoder().fit_transform(labels_with_noise) - 1
    # Compute NN distances of cluster members
    nn_arr = []
    for u_cl in np.unique(labels_traf[labels_traf > -1]):
        nn_arr.append(nn_data[labels_traf == u_cl].copy())

    cs, _ = remove_outlier_clusters(labels_traf, nn_arr, save_as_new_cluster=False)
    labels_clean = LabelEncoder().fit_transform(cs) - 1
    nb_remove_spurious_clusters = np.unique(labels_clean[labels_clean > -1]).size

    output_matrix[array_index, :] = labels_clean

    return nb_extracted_signal, nb_remove_spurious_clusters


def signal_spurious_simple(df_fit: pd.DataFrame, labels: np.array, te_obj: np.array, X_matrix: np.array,
                           output_matrix: np.array, array_index: int):
    """
    Older function that I built by merging the old extract_signal and remove_spurious_clusters functions from the
    jupyter-notebook I started out with.

    :param df_fit: Input dataframe that was provided to the SigMA clusterer instance
    :param labels: Label array
    :param density: clusterer.weights attribute
    :param X_matrix: clusterer.X attribute
    :param output_matrix: matrix or 1,N array to write the new labels to. Matrix style was coded for loops over
    multiple sf or KNN
    :param array_index: row index of the output matrix specifying the actual position in the loop
    :return: The numbers of found clusters after a) removing field stars and b) removing spurious clusters. Labels are
    written to the output matrix.
    """

    # very similar to extract_signal, except for the use of a different noise removal function
    labels_with_noise = -np.ones(df_fit.shape[0], dtype=int)
    data_idx = np.arange(df_fit.shape[0])
    for ii, u_cl in enumerate(np.unique(labels[labels > -1])):
        cluster_bool_array = remove_noise_simple(labels == u_cl, te_obj=te_obj)
    # if is_good_cluster:
    #     idx_cluster = data_idx[labels == u_cl][cluster_bool_array]
    #     labels_with_noise[idx_cluster] = ii

    nb_extracted_signal = np.unique(labels_with_noise[labels_with_noise > -1]).size

    # Compute nearest neighbors distances
    nn_data = nearest_neighbor_distribution(X_matrix)
    # Transform labels to start from 0 - (N-1)
    labels_traf = LabelEncoder().fit_transform(labels_with_noise) - 1
    # Compute NN distances of cluster members
    nn_arr = []
    for u_cl in np.unique(labels_traf[labels_traf > -1]):
        nn_arr.append(nn_data[labels_traf == u_cl].copy())

    cs, _ = remove_outlier_clusters(labels_traf, nn_arr, save_as_new_cluster=False)
    labels_clean = LabelEncoder().fit_transform(cs) - 1
    nb_remove_spurious_clusters = np.unique(labels_clean[labels_clean > -1]).size

    output_matrix[array_index, :] = labels_clean

    return nb_extracted_signal, nb_remove_spurious_clusters


def remove_field_stars(labels: np.array, density: np.array, output_matrix: np.array, array_index: int):
    """
    Function removing field stars from the clustering solution. Very lenient.

    :param labels: Label array
    :param density: clusterer.weights attribute
    :param output_matrix: matrix or 1,N array to write the new labels to. Matrix style was coded for loops over
    multiple sf or KNN
    :param array_index: row index of the output matrix specifying the actual position in the loop
    :return: The numbers of found clusters after a) removing field stars
    """

    lab = -np.ones_like(labels, dtype=np.int32)
    indices = np.arange(lab.size)
    for unique_cluster in np.unique(labels):
        indices_cluster = indices[labels == unique_cluster]
        lab[indices_cluster[remove_noise_quick_n_dirty(density, labels == unique_cluster)[0]]] = unique_cluster
    # Encode labels to range from 0 -- (N-1), for N clusters with field stars having "-1"
    labels_rfs = LabelEncoder().fit_transform(lab) - 1
    nb_remove_field_stars = np.unique(labels_rfs[labels_rfs > -1]).size

    # write row in the joint label array
    output_matrix[array_index, :] = labels_rfs

    return nb_remove_field_stars


def extract_signal(labels: np.ndarray, clusterer: object, output_matrix: np.array, array_index: int):
    """
    New function for extracting the signal (= clusters) from the clustering solution that Sebastian sent me
    in mid-October. Unlike the previous functions, it relies on the NoiseRemoval function remove_noise_simple.

    :param array_index: current row index of the output matrix to which labels are written
    :param output_matrix: Matrix holding the labels
    :param labels: array holding the labels determined by clusterer.fit()
    :param clusterer: SigMA instance applied to the dataset at hand
    :return: label array where field stars are denoted by -1, and the other groups by integers. ADDENDUM - I use the
    Label-Encoder to create cluster labels between 0 and N-1 for N found clusters
    """

    # initialize label array as all -1
    labels_with_noise = -np.ones(clusterer.X.shape[0], dtype=int)
    data_idx = np.arange(clusterer.X.shape[0])

    # iterate through all labels of cluster stars and remove noise with the custom function
    for i, u_cl in enumerate(np.unique(labels[labels > -1])):
        cluster_bool_array = remove_noise_simple(labels == u_cl, te_obj=clusterer)

        # make a distinction in case there is no noise to be removed (?)
        if cluster_bool_array is not None:
            labels_with_noise[cluster_bool_array] = i
        else:
            rho = clusterer.weights_[labels == u_cl]
            mad = np.median(np.abs(rho - np.median(rho)))
            threshold = np.median(rho) * 0.99 + 3 * mad * 1.2
            # Statistisch fundierterer cut
            # threshold = np.median(rho) + 3 * mad
            cluster_bool_array = rho > threshold
            idx_cluster = data_idx[labels == u_cl][cluster_bool_array]

            # I think this is invoked if more than 30 clusters are found
            if len(idx_cluster) > 30:
                # labels_with_noise[idx_cluster] = i
                # Only graph connected points allowed
                _, cc_idx = connected_components(clusterer.A[idx_cluster, :][:, idx_cluster])
                # Combine CCs data points with originally defined dense core
                # (to not miss out on potentially dropped points)
                cluster_indices = data_idx[idx_cluster][cc_idx == np.argmax(np.bincount(cc_idx))]
                labels_with_noise[np.isin(data_idx, cluster_indices)] = i

    labels_simple = LabelEncoder().fit_transform(labels_with_noise) - 1
    nb_simple = np.unique(labels_simple[labels_simple > -1]).size

    # write row in the joint label array
    output_matrix[array_index, :] = labels_simple

    return nb_simple


def consensus_function(label_matrix: np.array, density_sum: np.array, df_fit: pd.DataFrame, file: str = None,
                       path: str = None, plotting: bool = True):
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


def partial_clustering(region_label, df_input, sf_params, parameter_dict: dict, mode: str, output_loc: str,
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
        print(h, sol_group, "\n -----------------------------------")
        label_matrix = np.empty(shape=(len(sol_group), len(sol_group[0][1])))
        for k, entry in enumerate(sol_group):
            label_matrix[k, :] = entry[1]

        # Perform consensus clustering on the c) and d) steps
        labels_cc, n_cc = consensus_function(label_matrix, rho_sum, df_focus,
                                             f"{mode}_SimSol_{int(h)}_KNN_{knn}_CC",
                                             output_loc,
                                             plotting=True)

        grouped_label_matrix[h, :] = labels_cc

    labels_occ, n_occ = consensus_function(grouped_label_matrix, rho_sum, df_focus, f"{mode}_SimSol_KNN_{knn}_OCC",
                                           output_loc, plotting=True)

    return grouped_label_matrix, labels_occ, rho_sum
