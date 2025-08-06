import os
import pandas as pd
from sklearn.covariance import MinCovDet
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.metrics import f1_score
import DistantSigMA.misc.utilities as ut

matplotlib.use('TkAgg')

# ---------------------------------------------------------
# 1. Set output path + load data
# ---------------------------------------------------------

output_path = ut.set_output_path(script_name="DeNoise")

run = "Task_4_MCD"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load data
data = pd.read_csv(output_path + "SigMA_clustering_aligned.csv")
adj_matrix = load_npz(output_path + "adjacency_matrix.npz")

parameters = ["ra", "dec", "scaled_parallax", "pmra", "pmdec", "density"]

unique_labels = sorted(data.labels_SigMA_aligned.unique())

# support fractions to test: TESTED until 0.95 but the best was 0.75 max -> implemented new lims
supp = np.arange(0.5, 0.75, 0.05)

results = []
# for label in unique_labels[:2]:
for label in unique_labels:
    cluster = data[data["labels_SigMA_aligned"] == label]
    X = cluster[parameters].to_numpy()

    f1_scores = {}

    for s in supp:
        s = round(s, 2)
        data.loc[:, f"MCD_{s}"] = data['labels_SigMA_aligned']
        mcd = MinCovDet(support_fraction=s).fit(X)

        mcd_mask = mcd.support_
        # Get indices of the subset
        subset_idx = cluster.index

        # Create a mask over the full dataset with default True, set outliers to False
        mcd_mask_full = pd.Series(True, index=data.index)
        mcd_mask_full.loc[subset_idx] = mcd.support_

        condition = (data["labels_SigMA_aligned"] == label) & (~mcd_mask_full)
        data.loc[condition, f"MCD_{s}"] = -1

        # Now compute F1 score
        # Ground truth: 1 if from cluster i, else 0
        y_true = data['reference']  # adjust if your true labels column has a different name
        y_pred = data[f"MCD_{s}"]
        y_true_bin = (y_true == label).astype(int)

        # Prediction: 1 if predicted as cluster i (i.e., not -1), else 0
        y_pred_bin = (y_pred == label).astype(int)

        # Now compute F1
        try:
            score = f1_score(y_true_bin, y_pred_bin, average='binary')
            f1_scores[s] = score
        except ValueError:
            continue

    if f1_scores:
        best_s = max(f1_scores, key=f1_scores.get)
        best_score = f1_scores[best_s]
        worst_s = min(f1_scores, key=f1_scores.get)
        worst_score = f1_scores[worst_s]
        results.append({
            'cluster': label,
            'best_f1': best_score,
            'best_support': best_s,
            'worst_f1': worst_score,
            'worst_support': worst_s,
            'difference': best_score-worst_score,
        })


results_df = pd.DataFrame(results)

# Create scatterplot with Seaborn
f, ax = plt.subplots(3,1,figsize=(6, 6))
sns.scatterplot(
    data=results_df,
    x='cluster',
    y='best_f1',
    hue='best_support',
    palette='viridis',
    s=50,  # size of markers
    ax = ax[0],
)

sns.scatterplot(
    data=results_df,
    x='cluster',
    y='worst_f1',
    hue='worst_support',
    palette='viridis',
    s=50,
    ax = ax[1]# size of markers
)

sns.scatterplot(
    data=results_df,
    x='cluster',
    y='difference',
    hue='difference',
    palette='crest_r',
    s=50,
    ax = ax[2]# size of markers
)

plt.suptitle("Best F1 Scores per Cluster")
titles = ["Best", "Worst", "Difference"]
for ii, a in enumerate(ax):
    a.set_xlabel("Cluster Label")
    a.set_ylabel(titles[ii])
#plt.legend(title="Support Fraction", bbox_to_anchor=(1.05, 1), loc='upper left')
    a.set_xlim(-2,30)
    a.legend(loc="center right", bbox_to_anchor=(1.26, .5),)

ax[0].set_ylim(0.,1)
ax[1].set_ylim(0.,1)
ax[2].set_ylim(-0.01,0.15)
plt.subplots_adjust(right=0.8)

plt.savefig(output_path+"Support_fraction_test_clusters_facets_new_lims.png", dpi = 200)
plt.show()


# print(f"\nBest support fraction: {best_s} with F1 score: {best_score:.4f}")
#
# label_list = [f"MCD_{round(s, 2)}" for s in supp]
# label_list.insert(0, "reference")
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

# g.write_html(output_path + f"MCD_cluster_{label}_w_density_info.html")
