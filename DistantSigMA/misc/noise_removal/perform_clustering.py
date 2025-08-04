import os
import pandas as pd
from scipy.sparse import save_npz
from sklearn.metrics import normalized_mutual_info_score as nmi

from SigMA.SigMA import SigMA
from helpers import encode_labels_by_frequency, align_ref_to_sigma
from noise_removal_visualization import plot_3D_data

import DistantSigMA.misc.utilities as ut

# Paths
output_path = ut.set_output_path(script_name="DeNoise")

run = "Task_2_preRemoval_diagnostics"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

# mock_df = pd.read_csv("/Users/alena/PycharmProjects/Distant_SigMA/Data/Mock/Box_0_with_all_clustering_results.csv")
data = pd.read_csv("datafiles/Mock_clusters_newNoise.csv")

# Quality cuts -- obsolete because of the new input (01-08-25)
# data = mock_df.loc[mock_df['parallax'] > 0]
# data = data.loc[data['parallax'] / data['parallax_error'] > 4.5].reset_index(drop=True)

# ---------------------------------------------------------
# 2. Perform clustering with preset parameters
# ---------------------------------------------------------

xyz_axes = ['ra', 'dec', 'parallax']
pm_axes = ['pmra', 'pmdec']

# v_tan_axes = ['v_a_lsr', 'v_d_lsr']
# uvw_axes = ['U', 'V', 'W']

cluster_features = xyz_axes + pm_axes
scale_factors = {'vel': {'features': ['parallax'], 'factor': 0.2}}

# These are the default values and should be kept for now
sigma_kwargs = dict(
    cluster_features=cluster_features,
    scale_factors=scale_factors,
    nb_resampling=0, max_knn_density=101,
    beta=0.99, knn_initcluster_graph=45,
    do_remove_edges=True
)

clusterer = SigMA(data=data, **sigma_kwargs).fit(alpha=0.05, knn=15, bh_correction=True)

# ---------------------------------------------------------
# 3. Prepare result dataframe for DeNoise routine
# ---------------------------------------------------------

data.loc[:, "labels_SigMA"] = clusterer.labels_
data.loc[:, "density"] = clusterer.weights_  # Add density information
data.loc[:, "scaled_parallax"] = data["parallax"] * 0.2  # Add scaled parallax as column for MCD

# Create reference labels between -1 and (N-2)
data["reference"] = encode_labels_by_frequency(data["labels"])

# Align reference labels with closest matching SigMA labels
data_aligned = align_ref_to_sigma(input_data=data,
                                  ref_label="reference",
                                  label2match="labels_SigMA",
                                  new_col_name="labels_SigMA_aligned")

# sanity check for relabeling using NMI
print("NMI after clustering: ", nmi(data.labels, clusterer.labels_))
print("NMI after reference label encoding: ", nmi(data.reference, data.labels_SigMA))
print("NMI after label alignment: ", nmi(data_aligned.reference, data_aligned.labels_SigMA_aligned))

# ---------------------------------------------------------
# 4. Save dataframe and adjacency matrix (clusterer.A)
# ---------------------------------------------------------

data_aligned.to_csv(output_path+"SigMA_clustering_aligned.csv")
save_npz(output_path+'adjacency_matrix.npz', clusterer.A)

# fig_cart = plot_3D_data(data=data_aligned, labels=data_aligned.labels_SigMA_aligned,
#                         #xyz_axes=["ra", "dec", "scaled_parallax"],
#                         ax_range=[-300, 300]
#                         )
#
# fig_cart.write_html(output_path + "Mock_clusters_SigMA_raw_XYZ.html")