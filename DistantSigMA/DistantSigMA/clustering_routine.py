import pandas as pd
import numpy as np

from DistantSigMA.DistantSigMA.noise_removal_functions import setup_ICRS_ps, extract_signal_remove_spurious, \
    remove_field_stars, extract_signal, save_output_summary, consensus_function
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
