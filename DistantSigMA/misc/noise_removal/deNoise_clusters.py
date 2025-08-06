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

run = "Task_3_kneed_dynamic_ranges"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load data
data = pd.read_csv(output_path + "SigMA_clustering_aligned.csv")
adj_matrix = load_npz(output_path + "adjacency_matrix.npz")

# ---------------------------------------------------------
# 3. Add pruned data column (SigMA_noisy_pruned)
# ---------------------------------------------------------
# TODO: Does not work right now -- maybe fix later with extract signal function


# # ---------------------------------------------------------
# # 4. Perform DeNoise on all clusters (except Noise)
# # ---------------------------------------------------------
# TODO: Try concave for Nstars as right now it looks more like a knee with noise than an elbow
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

unique_labels = sorted(data.labels_SigMA_aligned.unique())

for label in sorted(unique_labels)[1:4]:
    obs_df, fig = deNoise_cluster(label=label,
                                  data=data,
                                  label_col="labels_SigMA_aligned",
                                  binning_strategy="equal",
                                  kneed_dicts=dicts,
                                  binsize=10)

    fig.savefig(output_path + f"Kneed_cluster_{label}.png", dpi=150)
    plt.close(fig)

    # label_list = ["reference", "labels_SigMA_aligned", "MCD_labels", "Nstar_labels", "velocity_labels",
    #               "volume_labels"]
    #
    # data["abs_G"] = data["phot_g_mean_mag"] + 5 * np.log10(data["parallax"]) - 10
    # data["bprp"] = data["phot_bp_mean_mag"] - data["phot_rp_mean_mag"]
    #
    # g = analyze_cluster(cluster_id=label,
    #                     df=data,
    #                     label_list=label_list,
    #                     cmd_params=["bprp", "abs_G"], pos_params=["ra", "dec", "scaled_parallax"],
    #                     vel_params=["pmra", "pmdec"])
    #
    # g.write_html(output_path + f"DeNoise_cluster_{label}.html")

    # observable_df.to_csv(f"datafiles/observables_cluster_{c_id}.csv", index=False)

    data.to_csv(output_path+f"labeled_deNoised_solution_for_cluster_{label}.csv")