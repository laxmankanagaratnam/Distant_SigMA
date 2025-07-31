import os
from itertools import islice
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from DeNoise import *
from helpers import analyze_solution, noisy_solution
from noise_removal_visualization import analyze_cluster
import DistantSigMA.misc.utilities as ut

matplotlib.use('TkAgg')

# ---------------------------------------------------------
# 1. Set output path + load data
# ---------------------------------------------------------

output_path = ut.set_output_path(script_name="DeNoise")

run = "DeNoise_with_pruning"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load data
input_data = pd.read_csv("datafiles/SigMA_clustering_aligned.csv")
adj_matrix = load_npz("datafiles/adjacency_matrix.npz")

# ---------------------------------------------------------
# 2. Artificially add noise to the labels
#
#    by assigning stars that were classified as -1 in both
#    the reference and SigMA labelling randomly to a given
#    cluster in the SigMA labels
# ---------------------------------------------------------

data = noisy_solution(p=0.5,
                      input_data=input_data,
                      reference_labels="reference",
                      clustered_labels="labels_SigMA_aligned",
                      new_label_col="SigMA_noisy")

data.sort_values(by="SigMA_noisy", inplace=True)

# ---------------------------------------------------------
# 3. Add pruned data column (SigMA_noisy_pruned)
# ---------------------------------------------------------
unique_labels = data.SigMA_noisy.unique()

data = prune_connected_components(df_in=data,
                                  label_col="SigMA_noisy",
                                  unique_labels=unique_labels,
                                  adjacency_matrix=adj_matrix,
                                  max_components_to_keep=1,
                                  )

# ---------------------------------------------------------
# 4. Perform DeNoise on all clusters (except Noise)
# ---------------------------------------------------------
# TODO: Think of special case for noise cluster to extract
#       cluster 28

data.to_csv("datafiles/SigMA_pruned.csv")

c_id = 1
label_col = "SigMA_noisy"
cluster = data[data[label_col] == c_id]

from sklearn.covariance import MinCovDet

parameters = ["ra", "dec", "scaled_parallax", "pmra", "pmdec"]

X = cluster[parameters].to_numpy()

mcd = MinCovDet(support_fraction=0.75).fit(X)

mcd_mask = mcd.support_
# Get indices of the subset
subset_idx = cluster.index

# Create a mask over the full dataset with default True, set outliers to False
mcd_mask_full = pd.Series(True, index=data.index)
mcd_mask_full.loc[subset_idx] = mcd.support_


data.loc[:, "MCD_support"] = data[label_col]
condition = (data[label_col] == c_id) & (~mcd_mask_full)
data.loc[condition, "MCD_support"] = -1




f, ax = plt.subplots(1, 4, figsize=(8, 3))

clips = [[0., 0.2], [0.1, 0.0], [0.2, 0.],[0., 0.5]]

dicts = [dict(curve="convex",
              direction="increasing",
              online=True,
              interp_method="polynomial",
              polynomial_degree=3,
              S=1),
         dict(curve="convex",
              direction="decreasing",
              online=True,
              interp_method="polynomial",
              polynomial_degree=3,
              S=1),
         dict(curve="convex",
              direction="decreasing",
              online=True,
              interp_method="polynomial",
              polynomial_degree=3,
              S=1),
         dict(curve="concave",
              direction="increasing",
              online=True,
              interp_method="polynomial",
              polynomial_degree=2,
              S=1)
         ]

observable_df = pd.DataFrame()
binsize = len(cluster) // 100
print("Binsize: ", binsize)
for m, method in enumerate(["Nstar", "velocity", "MCD", "volume"]):
    x, y = calculate_observables(method=method, cluster=cluster, binning="equal", binsize=binsize)

    # limit to the range of interest
    elbows = find_elbows(x=x, y=y, clip_fractions=clips[m], kneelocator_dict=dicts[m])

    print(elbows)
    if type(elbows) == list:
        elbows = elbows[0]

    data.loc[:, f"{method}_labels"] = data[label_col]
    condition = (data[label_col] == c_id) & (data.density < elbows)
    data.loc[condition, f"{method}_labels"] = -1

    axes = ax.ravel()
    axes[m].step(x, y, where='post', label=f"C {c_id}")
    axes[m].vlines(x=elbows, ymin=0, ymax=max(y), color="red", ls="dashed")
    axes[m].legend(loc="upper left")

    MCD_report, _ = analyze_solution(data=data, reference_labels="reference", labels2compare=f"{method}_labels")

    print(MCD_report.iloc[c_id+1])

    observable_df[f"{method}_x"] = x
    observable_df[f"{method}_y"] = y

f.show()

observable_df.to_csv(f"datafiles/observables_cluster_{c_id}.csv", index=False)

# Save results
#data.to_csv("datafiles/data_with_labels.csv", index=False)
#obs_df = pd.DataFrame(obs_records)
#obs_df.to_csv("datafiles/observables_long_format.csv", index=False)



#label_list = ["reference", "SigMA_noisy", "SigMA_noisy_pruned", "MCD_labels", "Nstar_labels", "velocity_labels", "volume_labels"]

label_list = ["reference","SigMA_noisy", "MCD_labels", "Nstar_labels", "MCD_support"]


data["abs_G"] = data["phot_g_mean_mag"] + 5 * np.log10(data["parallax"]) - 10
data["bprp"] = data["phot_bp_mean_mag"] - data["phot_rp_mean_mag"]

g = analyze_cluster(cluster_id=1,
                    df=data,
                    label_list=label_list,
                    cmd_params=["bprp", "abs_G"], pos_params=["ra", "dec", "scaled_parallax"],
                    vel_params=["pmra", "pmdec"])

g.write_html(output_path + f"DeNoise_cluster_{c_id}_pruned_MCDsupport.html")
