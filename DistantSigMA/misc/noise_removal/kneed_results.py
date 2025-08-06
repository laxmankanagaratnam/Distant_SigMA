from functools import reduce
import operator
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.covariance import MinCovDet
from astropy.stats import knuth_bin_width, freedman_bin_width
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import DistantSigMA.misc.utilities as ut

matplotlib.use('TkAgg')

# ---------------------------------------------------------
# 1. Set output path + load data
# ---------------------------------------------------------

output_path = ut.set_output_path(script_name="DeNoise")

run = "Task_3_kneed_analysis"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load data
label = 1
data = pd.read_csv(output_path + f"labeled_deNoised_solution_for_cluster_{label}.csv")


# Summary
# ---------------------------------------------------------------
def contamination_recall_analysis(df: pd.DataFrame, ref_label: str, label_list: list, cluster_id: int):
    fix_cluster_mask = (df[label_list[:]] == cluster_id).all(axis=1)
    fix_noise_mask = (df[label_list[:]] == -1).all(axis=1)

    true_mask = df[ref_label] == cluster_id

    for i in range(len(label_list) + 1):

        if i < len(label_list):
            label = label_list[i]
            pred_mask = df[label_list[i]] == cluster_id
        else:
            label = "intersection"

        TP = (true_mask & pred_mask).sum()
        FP = (~true_mask & pred_mask).sum()
        FN = (true_mask & ~pred_mask).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        contamination = 1 - precision

        # Subset data for this cluster in this strategy
        cluster_df = df[pred_mask]


cluster_mask = (data["labels_SigMA_aligned"] == label)

label_list = ["labels_SigMA_aligned", "MCD_labels", "Nstar_labels", "velocity_labels",
              "volume_labels"]

# Count number of times "label" appears in each column (i.e., each method)
label_counts = (data[label_list] == label).sum()

# Sort from most to least
sorted_label_counts = label_counts.sort_values(ascending=True)

# Get the sorted list of column names
sorted_label_list = sorted_label_counts.index.tolist()

data.loc[cluster_mask, 'N_detections'] = (data[sorted_label_list] == label).sum(axis=1)

cluster = data[data["labels_SigMA_aligned"] == label]
cluster = cluster.sort_values(by="density")

g = plt.figure(figsize=(4, 4))

for l in sorted_label_list:
    plt.plot(cluster['density'], cluster[l])

plt.show()

fix_cluster_mask = (cluster[sorted_label_list[:-1]] == label).all(axis=1)
fix_noise_mask = (cluster[sorted_label_list[:-1]] == -1).all(axis=1)
found_in_1 = (cluster[sorted_label_list[0]] == label)
found_in_2 = (cluster[sorted_label_list[1]] == label)
found_in_3 = (cluster[sorted_label_list[2]] == label)

def is_subset(subset_mask, superset_mask):
    return (subset_mask & ~superset_mask).sum() == 0


print("fix_cluster_mask ⊆ found_in_3:", is_subset(fix_cluster_mask, found_in_3))
print("found_in_2 ⊆ found_in_3:", is_subset(found_in_2, found_in_3))
print("found_in_1 ⊆ found_in_2:", is_subset(found_in_1, found_in_2))
"""
counts = {
    "Found in ≥4 (All but 1)": fix_cluster_mask.sum(),
    "Found in ≥3": found_in_3.sum() - fix_cluster_mask.sum(),
    "Found in ≥2": found_in_2.sum() - found_in_3.sum(),
    "Found in ≥1": found_in_1.sum() - found_in_2.sum(),
    "Found in 0 (Noise)": fix_noise_mask.sum()
}

labels = list(counts.keys())
values = list(counts.values())

plt.figure(figsize=(8, 6))
bars = plt.barh(labels, values)
plt.xlabel("Number of Sources")
plt.title("Detection Overlap Across Methods")
plt.tight_layout()
plt.show()
"""

