# Python modules
import os
import numpy as np
import pandas as pd

# SigMA modules
from SigMA.bayesian_velocity_scaling import scale_factors as sf_function

# DistantSigMA modules
from DistantSigMA.DistantSigMA.clustering_routine import run_clustering_cartesian
from DistantSigMA.DistantSigMA.PlotlyResults import plot

# --------- Paths & Setup ---------
# TODO: Change output path to your preferred location
output_path = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Sigma_Orion/Coding/Code_output/2025-11-03/ScoCen_pipeline_test/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ------------- Data --------------
# TODO: Change to your pre-processed file
df = pd.read_csv("../ISM-FLOW-WS2/Vela_clusters_DR3_preprocessed.csv")

print(df.shape)

df_focus = df.dropna(subset=["X","Y","Z","v_a_lsr", "v_d_lsr"])

print(df_focus.shape)

# ===================================
#            Clustering
# ===================================

# 1. User-defined parameters
# FIXME: Change parameters here
KNNs = [15, 30]
bh_corr = True
alpha = 0.05  # other option: 0.01

# 2. Create the parameter dict for SigMA clusterer -- these values stay fixed
run_dict = dict(alpha=alpha,
                beta=0.99,
                knn_initcluster_graph=35,
                KNN_list=KNNs,
                bh_correction=bh_corr,
               kd_tree_data=None
                )

# 3.Run Clustering

df_CC = run_clustering_cartesian(df_input=df_focus, parameter_dict=run_dict, nb_res=0,
                                 bayesian_file_path="bayesian_LR_data.npz",
                                 output_loc=output_path)
