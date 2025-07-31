import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.covariance import MinCovDet

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.stats import knuth_bin_width, freedman_bin_width
from kneed import KneeLocator

import matplotlib

matplotlib.use('TkAgg')  # or 'QtAgg', depending on what you have installed

from helpers import analyze_solution, noisy_solution

input_data = pd.read_csv(
    "/DistantSigMA/misc/noise_removal/datafiles/noise_removal_input_aligned.csv",
)

input_data.loc[:, "scaled_parallax"] = input_data["parallax"] * 0.2

data = noisy_solution(p=0.5,
                      input_data=input_data,
                      reference_labels="real_ref",
                      clustered_labels="labels_SigMA_aligned",
                      new_label_col="SigMA_noisy")

data.sort_values(by="SigMA_noisy", inplace=True)

# ------------- Stats check --------
# print("NMI: ", nmi(input_data.labels, input_data.labels_SigMA_aligned))
# report, fig = analyze_solution(data=input_data, reference_labels="real_ref", labels2compare="labels_SigMA_aligned")
# print(report)
# fig.show()
# report, fig2 = analyze_solution(data=data, reference_labels="real_ref", labels2compare="SigMA_noisy")
# fig2.show()

print("NMI: ", round(nmi(data.labels, data.SigMA_noisy), 3))


def calculate_MCD_determinant(sample, parameters=None):
    # Simulated data: 100 samples, 5 features
    if parameters is None:
        parameters = ["ra", "dec", "scaled_parallax", "pmra", "pmdec"]

    X = sample[parameters].to_numpy()

    mcd = MinCovDet(support_fraction=0.8).fit(X)

    mean = mcd.location_  # shape: (5,)
    covariance = mcd.covariance_  # shape: (5, 5)
    support = mcd.support_

    det = np.linalg.det(mcd.covariance_)

    return det


f, ax = plt.subplots(5, 6, figsize=(10, 10))
axes = ax.ravel()

data.loc[:, "MCD_labels"]= data.SigMA_noisy


for l, label in enumerate(sorted(data.SigMA_noisy.unique()[1:2])):

    # grab cluster
    cluster = data[data.SigMA_noisy == label]
    cluster_sorted = cluster.sort_values(by="density")
    sorted_densities = cluster_sorted.density.to_numpy()

    # calculate determinant for found cluster
    det_found = calculate_MCD_determinant(sample=cluster)
    print(f"Determinant of found Cluster {label}: {det_found}")

    # calculate determinant for true cluster
    det_true = calculate_MCD_determinant(sample=data[data["real_ref"] == label])
    print(f"True Determinant of Cluster {label}: {det_true}")

    min_density = sorted_densities[0] + sorted_densities[0] * 0.1
    max_density = sorted_densities[-1] - sorted_densities[-1] * 0.5

    # Create a mask for densities within desired range
    mask = (sorted_densities >= min_density) & (sorted_densities <= max_density)

    # Filter arrays accordingly
    filtered_densities = sorted_densities[mask]

    # Compute bin width and bin edges
    bin_width, bins = freedman_bin_width(filtered_densities, return_bins=True)

    # Compute histogram using the bins
    hist_counts, bin_edges = np.histogram(filtered_densities, bins=bins)

    # Compute cumulative counts
    cumulative_counts = np.cumsum(hist_counts)

    # caluclate determinants
    results = []
    p = cluster.shape[1]  # number of parameters/features (needed for a sanity check)

    for i, (cum_n, right) in enumerate(zip(cumulative_counts, bin_edges[1:])):
        # cumulative subset: all points with density <= right
        subset = cluster_sorted[cluster_sorted["density"] <= right]

        # (optional) safety: MCD needs at least p+1 observations to get a full scatter matrix
        if subset.shape[0] <= 5:
            det = np.nan
        else:
            det = calculate_MCD_determinant(sample=subset)

        results.append(det)

    dets = np.array(results)

    # For plotting (optional): get bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    kl = KneeLocator(x=bin_centers, y = dets, curve="convex", direction="decreasing", online=True, interp_method="polynomial", polynomial_degree=3, S =1)
    min_elbow = min(kl.all_elbows)
    max_elbow = max(kl.all_elbows)
    print(min_elbow)
    print(max_elbow)

    axes[label].step(bin_centers, dets, where='post', label=f"C {label}")
    axes[label].hlines(y=det_true, xmin=0, xmax=max(sorted_densities), color="black", lw=0.5, )
    axes[label].vlines(x=min_elbow, ymin =0, ymax=max(dets), color="red", ls="dashed")
    axes[label].vlines(x=max_elbow, ymin =0, ymax=max(dets), color="green", ls="dashed")
    axes[label].legend(loc="upper left")


    # Check performance
    star_counts = (sorted_densities >= max_elbow).sum()
    true_cluster =  data[data.real_ref == label]
    print(f"{label}: {star_counts}/{len(cluster)} (true: {len(true_cluster)})")

    condition = (data.SigMA_noisy == label) & (data.density < max_elbow)
    data.loc[condition, "MCD_labels"] = -1

plt.show()


MCD_report, fig3 = analyze_solution(data=data, reference_labels="real_ref", labels2compare="MCD_labels")
fig3.show()

data.to_csv("/Users/alena/PycharmProjects/Distant_SigMA/DistantSigMA/misc/noise_removal/NR__MCD.csv")
