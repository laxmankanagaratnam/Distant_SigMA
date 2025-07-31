from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.covariance import MinCovDet
from astropy.stats import knuth_bin_width, freedman_bin_width
from scipy.spatial import Delaunay


# ---------------------------------------------------------------
# Observable calculation functions
# ---------------------------------------------------------------

def tangential_velocity_variance(df):
    v_x = df.v_a_lsr.to_numpy()
    v_y = df.v_d_lsr.to_numpy()

    return np.var(v_x) + np.var(v_y)


def MCD_determinant(sample, parameters=None, support_fraction=.99):
    # Simulated data: 100 samples, 5 features
    if parameters is None:
        parameters = ["ra", "dec", "scaled_parallax", "pmra", "pmdec"]

    X = sample[parameters].to_numpy()

    mcd = MinCovDet(support_fraction=support_fraction).fit(X)

    # mean = mcd.location_  # shape: (5,)
    # covariance = mcd.covariance_  # shape: (5, 5)
    # support = mcd.support_

    det = np.linalg.det(mcd.covariance_)

    return det


def MCD_determinant_raw(sample, parameters=None, support_fraction=.5):
    # Simulated data: 100 samples, 5 features
    if parameters is None:
        parameters = ["ra", "dec", "scaled_parallax", "pmra", "pmdec"]

    X = sample[parameters].to_numpy()

    mcd = MinCovDet(support_fraction=support_fraction).fit(X)

    # mean = mcd.location_  # shape: (5,)
    # covariance = mcd.covariance_  # shape: (5, 5)
    # support = mcd.support_

    raw_det = np.linalg.det(mcd.raw_covariance_)

    return raw_det


def volume_from_delaunay(points, parameters=None):
    if parameters is None:
        parameters = ["ra", "dec", "scaled_parallax"]

    X = points[parameters].to_numpy()

    if X.shape[0] < 4:
        return 0.0  # Not enough points to form a volume

    try:
        delaunay = Delaunay(X)
    except Exception as e:
        print(f"Delaunay triangulation failed: {e}")
        return np.nan

    simplices = X[delaunay.simplices]  # shape (n_simplices, 4, 3)

    # Volume of tetrahedron: |det([a−d, b−d, c−d])| / 6
    def tetra_volume(tetra):
        a, b, c, d = tetra
        return abs(np.linalg.det(np.stack([a - d, b - d, c - d]))) / 6

    volumes = np.array([tetra_volume(tetra) for tetra in simplices])
    return np.sum(volumes)


# ---------------------------------------------------------------
# Density binning
# ---------------------------------------------------------------


def create_auto_bins(method: str, x_data):
    if method == "freedman":
        _, bin_edges = freedman_bin_width(x_data, return_bins=True)
    elif method == "knuth":
        _, bin_edges = knuth_bin_width(x_data, return_bins=True)
    else:
        raise ValueError("WARNING: Check input method -- only 'freedman' or 'knuth' are accepted as arguments!")

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, bin_edges


def calculate_density_ranges(cluster_sorted: pd.DataFrame, binning: str, binsize: int = None):
    # define these here in case of no binning
    density_array = cluster_sorted.density.to_numpy()
    bin_edges = None

    if binning in ("freedman", "knuth"):
        density_array, bin_edges = create_auto_bins(binning, density_array)

    elif binning == "rolling":
        density_array = []
        bin_edges = []

        for i in range(len(cluster_sorted) - binsize + 1):
            window = cluster_sorted.iloc[i:i + binsize]
            mean_density = window['density'].mean()
            density_array.append(mean_density)
            bin_edges.append(window['density'].iloc[-1])  # last density in the window

        density_array = np.array(density_array)
        bin_edges = np.array(bin_edges)

    elif binning == "equal":
        n = len(cluster_sorted)
        num_bins = (n + binsize - 1) // binsize  # ceil division to include last short bin
        bin_edges = []

        for i in range(1, num_bins + 1):
            end_idx = min(i * binsize, n)
            edge_value = cluster_sorted['density'].iloc[end_idx - 1]
            bin_edges.append(edge_value)

        bin_edges = np.array(bin_edges)

        df = cluster_sorted.reset_index(drop=True)
        df['group'] = df.index // binsize
        density_array = df.groupby('group')['density'].mean().to_numpy()
        print(len(density_array))
    return density_array, bin_edges


# ---------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------
def prune_connected_components(
        df_in,
        label_col,
        unique_labels,
        adjacency_matrix,
        max_components_to_keep=1,
        new_col_name: str = None,
):
    if new_col_name is None:
        new_col_name = f"{label_col}_pruned"
    df = df_in.copy()
    df[new_col_name] = df[label_col]  # initialize once

    for lbl in unique_labels:
        cluster = df[df[label_col] == lbl]
        indices = cluster.index.to_list()

        A_sub = adjacency_matrix[np.ix_(indices, indices)]
        graph_sub = csr_matrix(A_sub)
        n_components, c_lab = connected_components(graph_sub, directed=False, return_labels=True)

        if n_components == 0:
            continue

        component_labels = np.array(c_lab)
        unique_components, counts = np.unique(component_labels, return_counts=True)

        # Filter out components of size 1
        large_components = [(comp, count) for comp, count in zip(unique_components, counts) if count > 1]

        # If no large components, skip
        if len(large_components) == 0:
            mask_to_update = (df[label_col] == lbl)
            df.loc[mask_to_update, new_col_name] = -1
            print(f"Label {lbl} — all components too small (size 1), everything pruned.")
            continue

        # Sort by size descending
        large_components.sort(key=lambda x: x[1], reverse=True)

        # Keep up to N largest
        top_components = [comp for comp, _ in large_components[:max_components_to_keep]]

        # Collect indices to keep
        indices_to_keep = set()
        for comp_id in top_components:
            member_positions = np.where(component_labels == comp_id)[0]
            indices_to_keep.update(indices[i] for i in member_positions)

        # Prune all others
        mask_to_update = (df[label_col] == lbl) & (~df.index.isin(indices_to_keep))
        df.loc[mask_to_update, new_col_name] = -1

        # print(f"Label {lbl} — Original: {len(indices)}, Kept: {len(indices_to_keep)}, Pruned: {np.sum(mask_to_update)}")

    return df


# ---------------------------------------------------------------
# Elbow Detection
# ---------------------------------------------------------------
def find_elbows(x: np.array, y: np.array, clip_fractions: list, kneelocator_dict: dict):
    # limit to the range of interest
    min_x = x[0] + x[0] * clip_fractions[0]
    max_x = x[-1] - x[-1] * clip_fractions[1]
    mask = (x >= min_x) & (x <= max_x)
    clipped_x = x[mask]
    clipped_y = y[mask]

    kl = KneeLocator(x=clipped_x, y=clipped_y, **kneelocator_dict)

    if len(kl.all_elbows) == 0:
        print("No elbows found.")
        elbow = np.nan
    elif len(kl.all_elbows) == 1:
        print("One elbow was found.")
        elbow = min(kl.all_elbows)
    else:
        print("More than one elbow found. Returning min/max values.")
        elbow = [min(kl.all_elbows), max(kl.all_elbows)]

    return elbow


# ---------------------------------------------------------------
# Observable Evaluator
# ---------------------------------------------------------------

def calculate_observables(method: str, cluster: pd.DataFrame, binning: str, binsize: int = None):
    function_dict = {
        "Nstar": None,
        "velocity": tangential_velocity_variance,
        "MCD": MCD_determinant,
        "MCD_raw": MCD_determinant_raw,
        "volume": volume_from_delaunay,

    }

    function = function_dict[method]

    cluster_sorted = cluster.sort_values(by="density").reset_index(drop=True)

    _, x_vals = calculate_density_ranges(cluster_sorted, binning,
                                         binsize)  # NOTE: The X vals are the cumulative bin edges
    if method == "Nstar":
        y_vals = np.searchsorted(cluster_sorted.density, x_vals, side='right')  # TODO: This is likely not correct

    else:

        results = []
        for edge in x_vals:
            subset = cluster_sorted[cluster_sorted.density <= edge]  # cumulative subset
            if len(subset) < 6:
                results.append(np.nan)  # or skip/append 0/etc depending on your logic
            else:
                res = function(subset)
                results.append(res)

        y_vals = np.array(results)
    return x_vals, y_vals


def deNoise_cluster(label, data, label_col, binning_strategy: str, kneed_dicts: dict,
                    n_per_bin: int,
                    plot: bool = True,
                    methods=None,
                    ):
    if methods is None:
        methods = ["Nstar", "velocity", "MCD", "volume"]

    cluster = data[data[label_col] == label]
    print(f"\nProcessing cluster {label} (size={len(cluster)})")
    binsize = len(cluster) // n_per_bin

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        axes = ax.ravel()
    else:
        fig, axes = np.nan, np.nan

    for m, method in enumerate(methods):

        x, y = calculate_observables(method=method, cluster=cluster, binning=binning_strategy, binsize=binsize)

        elbows = find_elbows(x=x, y=y, clip_fractions=[0.5, 0.1], kneelocator_dict=kneed_dicts[method])

        if isinstance(elbows, list):
            elbows = elbows[0]
        print(f"{method} elbow: {elbows}")

        # Update data: assign -1 to below-elbow densities for this cluster
        method_label_col = f"{method}_labels"

        mask_noise = (data[label_col] == label) & (data.density < elbows)
        mask_cluster = (data[label_col] == label)

        data.loc[mask_noise, method_label_col] = -1
        data.loc[mask_cluster, f"{method}_x"] = x
        data.loc[mask_cluster, f"{method}_y"] = y

        if plot:
            # Plot
            axes[m].step(x, y, where='post', label=f"Cluster {label}")
            axes[m].vlines(x=elbows, ymin=0, ymax=max(y), color="red", ls="dashed")
            axes[m].legend(loc="upper left")

    return fig


if __name__ == "__main__":

    # kneelocator dictionaries differ depending on the observable
    kneelocator_dicts = {
        "Nstar": dict(curve="convex", direction="increasing",
                      online=True, interp_method="polynomial",
                      polynomial_degree=3, S=1),
        "velocity": dict(curve="convex", direction="decreasing",
                         online=True, interp_method="polynomial",
                         polynomial_degree=3, S=1),
        "MCD": dict(curve="convex", direction="decreasing",
                    online=True, interp_method="polynomial",
                    polynomial_degree=3, S=1),
        "volume": dict(curve="concave", direction="increasing",
                       online=True, interp_method="polynomial",
                       polynomial_degree=2, S=1)
    }

    methods = ["Nstar", "velocity", "MCD", "volume"]

    # copy dataframe for good measure
    data_deNoise = data.copy()
    for method in methods:
        data_deNoise.loc[:, f"{method}_label"] = data_deNoise.SigMA_noisy
        data_deNoise.loc[:, f"{method}_x"] = 0
        data_deNoise.loc[:, f"{method}_y"] = 0

    obs_records = []  # for long-format observable curves

    for label, cluster in islice(data.groupby("SigMA_noisy"), 1, 4):
        fig = deNoise_cluster(label, data, "SigMA_noisy", "equal", kneelocator_dicts, 100)
        plt.show()
