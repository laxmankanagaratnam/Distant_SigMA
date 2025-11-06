# Python modules
import os
import numpy as np
import pandas as pd

# DistantSigMA modules
from DistantSigMA.DistantSigMA.clustering_routine import run_clustering_ICRS

# --------- Paths & Setup ---------
# TODO: Change output path to your preferred location -> Do not forget to add a / at the end
output_path = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Sigma_Orion/Coding/Code_output/2025-11-06/Vela_test/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ------------- Data --------------
# TODO: Change to your file
df = pd.read_csv("../ISM-FLOW-WS2/Vela_clusters_DR3_preprocessed.csv")

print(df.shape)

# Clean the dataset
df_focus = df.dropna(subset=["ra", "dec", "parallax", "pmra", "pmdec"])

print(df_focus.shape)

# ===================================
#            Clustering
# ===================================

# 1. User-defined parameters
# FIXME: Change parameters here
KNNs = [15, 30]
bh_corr = True

# 2. Create the parameter dict for SigMA clusterer -- these values stay fixed
run_dict = dict(KNN_list=KNNs,
                bh_correction = bh_corr,
                alpha=0.05,
                beta=0.99,
                knn_initcluster_graph=35,
                sfs=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
                scaling="robust",
                kd_tree_data=df_focus[['ra', 'dec', 'parallax', 'pmra', 'pmdec']]
                )

# 3.Run Clustering
df_CC = run_clustering_ICRS(df_input=df_focus,
                            parameter_dict=run_dict,
                            noise_removal="strict",
                            output_loc=output_path)
print(df_CC.shape)

df_clusters = df_CC[df_CC.cluster_label != -1]


print(df_clusters.shape)
