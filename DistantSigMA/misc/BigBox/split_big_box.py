import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from SigMA.SigMA import SigMA
from coordinate_transformations.sky_convert import transform_sphere_to_cartesian

from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility
from DistantSigMA.DistantSigMA.setup_and_scaling import setup_Cartesian_ps
from DistantSigMA.DistantSigMA.coarse_clustering import get_segments, merge_subsets
from DistantSigMA.DistantSigMA.PlotlyResults import plot_darkmode


# ---------------------------------------------------------
# Set output directory and path
script_name = my_utility.get_calling_script_name(__file__)
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/', script_name=script_name)

# ---------------------------------------------------------
# Load data
df_focus = pd.read_csv("/Users/alena/PycharmProjects/Distant_SigMA/Data/Gaia/OrionRectangle_plx_oerror_bigger_4p5_fidelity_bigger_0p5.csv")

# ---------------------------------------------------------
# Setup and scaling
beta = 0.99
knn_initcluster_graph = 35
KNNs = range(550,1050, 50)
alpha = 0.05
scaling = "bayesian"
bh_correction = True

# setup kwargs
setup_kwargs = setup_Cartesian_ps(df_fit=df_focus, KNN_list=KNNs, beta=beta,
                                  knn_initcluster_graph=knn_initcluster_graph,
                                  info_path="/Users/alena/PycharmProjects/Distant_SigMA/Data/bayesian_LR_data.npz")

sigma_kwargs = setup_kwargs["sigma_kwargs"]
scale_factor_list = setup_kwargs["scale_factor_list"][:]

sigma_kwargs["transform_function"] = transform_sphere_to_cartesian

print(sigma_kwargs, scale_factor_list)


# ---------------------------------------------------------
# Clustering - not noise removal

clusterer_coarse = SigMA(data=df_focus, **sigma_kwargs)  # initialize SigMA with sf_mean

meta_df = pd.DataFrame(columns=["KNN", "n_clusters", "n_stars"])

for k,knn in enumerate(KNNs):  # Single result for each KNN

    rho_sum = np.zeros(df_focus.shape[0], dtype=np.float32)      # initialize the density sum
    label_matrix_coarse = np.empty(shape=(len(scale_factor_list), len(df_focus)))      # initialize the label matrix for the different scaling factors

    for sf_id, sf in enumerate(scale_factor_list[:]):  # Loop over scale factors

        scale_factors = {'vel': {'features': ['v_a_lsr', 'v_d_lsr'], 'factor': sf}}
        clusterer_coarse.set_scaling_factors(scale_factors)  # Set new scale factor
        print(f"Performing clustering for scale factor {clusterer_coarse.scale_factors}...")

        # Fit
        clusterer_coarse.fit(alpha=alpha, knn=knn, bh_correction=bh_correction)
        label_array = clusterer_coarse.labels_

        # raw labels
        labels_real = LabelEncoder().fit_transform(label_array)  # Transform to 0 - (N-1)
        label_matrix_coarse[sf_id, :] = labels_real

        # density
        rho = clusterer_coarse.weights_
        rho_sum += rho
        
        # Plot to verify
        df_verify = df_focus.copy()
        df_verify["labels"] = labels_real
       # plot_darkmode(labels=df_verify["labels"], df=df_verify, filename=f"solution_sf_{sf_id}_{round(sf,2)}",
       #               output_pathname=output_path)
    # Get consensus
    combined_pred = get_segments(df_focus, sigma_kwargs["cluster_features"], label_matrix_coarse, verify_results_path=output_path)
    df_save = merge_subsets(df_focus, combined_pred, knn)    # re-merge smallest regions if their size is smaller than KNN value
    #df_save = df_focus.copy()

    unique_labels, counts = np.unique(combined_pred, return_counts=True)
    meta_df.loc[k, "KNN"] = knn
    meta_df.loc[k, "n_clusters"] = len(unique_labels)
    meta_df.loc[k, "n_stars"] = counts

    # ---------------------------------------------------------
    # output path extension for runs
    run = f"Big_box_KNN_{knn}"

    if not os.path.exists(output_path + f"Run_{run}/"):
        os.makedirs(output_path + f"Run_{run}/")

    result_path = output_path + f"Run_{run}/"

    # plotting
    plot_darkmode(labels=df_save["chunk_labels"], df=df_save, filename=f"RF_run_{run}",
          output_pathname=result_path)

    # save dataframe for subsequent runs on the separate regions
    df_save.to_csv(result_path + f"RF_run_{run}.csv")
    # df_labels.to_csv(result_path+f"RF_run_{run}_all_sf_labels.csv")

    # Output log-file
    all_fixed = {"mode": "Cartesian", "alpha": alpha, "beta": beta, "knn_initcluster_graph": knn_initcluster_graph,
                 "KNN_list": KNNs, "bh_correction": bh_correction, "sfs_list": scale_factor_list,
                 "scaling": scaling,"nb_resampling": sigma_kwargs["nb_resampling"]}

    filename = result_path + f"Run_{run}_parameters.txt"
    with open(filename, 'w') as file:
        for key, value in all_fixed.items():
            file.write(f"{key} = {value}\n")

meta_df.to_csv(output_path+"metadata.csv")
print("--------- Routine executed successfully ---------")
