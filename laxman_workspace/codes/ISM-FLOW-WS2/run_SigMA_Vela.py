# Python modules
import os
import sys
import pathlib
import numpy as np
import pandas as pd

# Setup path to find DistantSigMA modules
repo_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# DistantSigMA modules
from DistantSigMA.DistantSigMA.clustering_routine import run_clustering_ICRS

# --------- Paths & Setup ---------
# Get the script directory
script_dir = pathlib.Path(__file__).parent

# Input path to clustering results (relative to script location)
input_file = script_dir / "vela_unknown_clustering_results.csv"

# Output path
output_path = str(script_dir / "output" / "unknown" / "")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ------------- Data --------------
# Load the vela_unknown clustering results
df = pd.read_csv(input_file)

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

# 4. Save results to CSV
output_csv = os.path.join(output_path, "vela_unknown_clustering_complete.csv")
df_CC.to_csv(output_csv, index=False)
print(f"\nClustering results saved to: {output_csv}")

# Save only clustered stars to separate CSV
output_csv_clusters = os.path.join(output_path, "vela_unknown_clusters_only.csv")
df_clusters.to_csv(output_csv_clusters, index=False)
print(f"Clustered stars only saved to: {output_csv_clusters}")
