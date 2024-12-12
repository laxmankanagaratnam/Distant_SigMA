# Python modules
import os
from DistantSigMA.misc import utilities as ut
from DistantSigMA.DistantSigMA.clustering_routine import *
from DistantSigMA.DistantSigMA.scalefactor_sampling import lhc_lloyd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def plot_darkmode(labels, df: pd.DataFrame, filename: str, output_pathname: str = None, hrd: bool = False,
                  icrs: bool = False, return_fig: bool = False):
    """ Simple function for creating a result plot of all the final clusters in Dark mode."""

    # not relevant for the end result
    if icrs:
        vel1 = "pmra"
        vel2 = "pmdec"
    else:
        vel1 = "v_a_lsr"
        vel2 = "v_d_lsr"

    clustering_solution = labels  # set label variable
    df_plot = df
    # plotting specs
    bg_opacity = 0.1
    bg_color = 'gray'
    plt_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#FF6692', '#B6E880', '#FF97FF',
                  '#FECB52', '#B82E2E', '#316395']

    #  ---------------  Create figure  ---------------
    #  --------------- ---------------  --------------
    # without HRD subplot
    if not hrd:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
            column_widths=[0.7, 0.3],
            subplot_titles=['position', 'velocity'], )

    # with HRD subplot (slower if there are many clusters)
    else:
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]],
            column_widths=[0.4, 0.3, 0.3],
            subplot_titles=['3D positions', 'Velocities', 'HRD'], )

    # --------------- 3D scatter plot -------------------
    # background
    trace_3d_bg = go.Scatter3d(
        x=df_plot.loc[:, 'X'], y=df_plot.loc[:, 'Y'], z=df_plot.loc[:, 'Z'],
        mode='markers', marker=dict(size=1, color=bg_color, opacity=bg_opacity), hoverinfo='none', showlegend=False, )
    fig.add_trace(trace_3d_bg, row=1, col=1)

    # sun marker
    trace_sun = go.Scatter3d(
        x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
        mode='markers', marker=dict(size=5, color='red', symbol='x'), hoverinfo='none', showlegend=True, name='Sun')
    fig.add_trace(trace_sun, row=1, col=1)

    # 3D cluster
    for j, uid in enumerate(np.unique(clustering_solution)):
        if uid != -1:
            plot_points = (clustering_solution == uid)  # grab the right locations
            trace_3d = go.Scatter3d(
                x=df_plot.loc[plot_points, 'X'], y=df_plot.loc[plot_points, 'Y'], z=df_plot.loc[plot_points, 'Z'],
                mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)]), hoverinfo='none',
                showlegend=True, name=f'Cluster {uid} ({np.sum(plot_points)} stars)', legendgroup=f'group-{uid}' )
            fig.add_trace(trace_3d, row=1, col=1)  # add cluster trace

    # --------------- 2D vel plot -------------------

    # background
    trace_vel_bg = go.Scatter(
        x=df_plot.loc[:, vel1], y=df_plot.loc[:, vel2],
        mode='markers', marker=dict(size=3, color=bg_color, opacity=bg_opacity), hoverinfo='none', showlegend=False)
    fig.add_trace(trace_vel_bg, row=1, col=2)

    # cluster velocities (same as for 3D positions)
    for j, uid in enumerate(np.unique(clustering_solution)):
        if uid != -1:
            plot_points = (clustering_solution == uid)  # & cut_us
            trace_vel = go.Scatter(x=df_plot.loc[plot_points, vel1], y=df_plot.loc[plot_points, vel2],
                                   mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)], ),
                                   hoverinfo='none', legendgroup=f'group-{uid}',
                                   name=f'Cluster {uid} ({np.sum(plot_points)} stars)', showlegend=False)
            fig.add_trace(trace_vel, row=1, col=2)

    # ------------ Update axis information ---------------
    # 3d position
    plt_kwargs = dict(showbackground=False, showline=False, zeroline=True, zerolinecolor='white', zerolinewidth=2,
                      showgrid=True, showticklabels=True, color="white",
                      linecolor='white', linewidth=1, gridcolor='white')

    xaxis = dict(**plt_kwargs, title='X [pc]')  # , tickmode = 'linear', dtick = 50, range=[-50,200])
    yaxis = dict(**plt_kwargs, title='Y [pc]')  # , tickmode = 'linear', dtick = 50, range=[-200, 50])
    zaxis = dict(**plt_kwargs, title='Z [pc]')  # , tickmode = 'linear', dtick = 50, range=[-100, 150])

    # tangential vel
    if not icrs:
        fig.update_xaxes(title_text="v_alpha", showgrid=False, row=1, col=2, color="white")
        fig.update_yaxes(title_text="v_delta", showgrid=False, row=1, col=2, color="white")
    else:
        fig.update_xaxes(title_text="pmra", showgrid=False, row=1, col=2, color="white")
        fig.update_yaxes(title_text="pmdec", showgrid=False, row=1, col=2, color="white")
    # tangential vel

    # Finalize layout
    fig.update_layout(
        title="",
        # width=800,
        # height=800,
        showlegend=True,
        paper_bgcolor='#383838',
        plot_bgcolor='#383838',
        legend=dict(itemsizing='constant',  font=dict(
            color="white"  # Set legend text color to white
        )),
        # 3D plot
        scene=dict(
            xaxis=dict(xaxis),
            yaxis=dict(yaxis),
            zaxis=dict(zaxis)
        )
    )

    if output_pathname:
        fig.write_html(output_pathname + f"{filename}.html")

    if return_fig:
        return fig


def compute_centroids(labels, data_points):
    centroids = {}
    centroid_data = data_points[['ra', 'dec', 'parallax', 'pmra', 'pmdec']].values
    for cluster_id in np.unique(labels):
        cluster_points = centroid_data[labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        centroids[cluster_id] = centroid
    return centroids


def relabel_clusters(prev_centroids, prev_counts, current_centroids, current_counts, current_labels, threshold=0.5):
    # Prepare data for cost matrix
    prev_labels = np.array(list(prev_centroids.keys()))
    current_labels_unique = np.array(list(current_centroids.keys()))
    prev_centroid_array = np.array([prev_centroids[label] for label in prev_labels])
    current_centroid_array = np.array([current_centroids[label] for label in current_labels_unique])
    prev_count_array = np.array([prev_counts[label] for label in prev_labels])
    current_count_array = np.array([current_counts[label] for label in current_labels_unique])

    # Compute centroid and count distances
    centroid_distances = cdist(prev_centroid_array, current_centroid_array)
    count_differences = np.abs(prev_count_array[:, None] - current_count_array)

    # Normalize distances
    centroid_distances /= centroid_distances.max()
    count_differences = count_differences / count_differences.max()

    # Combine into cost matrix
    cost_matrix = centroid_distances + count_differences

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Initialize mapping and split tracking
    new_label_mapping = {}
    split_clusters = {}

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < threshold:  # Match found
            if prev_labels[r] in new_label_mapping:
                # Handle a split
                if prev_labels[r] not in split_clusters:
                    split_clusters[prev_labels[r]] = [new_label_mapping[prev_labels[r]]]
                split_clusters[prev_labels[r]].append(current_labels_unique[c])
            else:
                new_label_mapping[current_labels_unique[c]] = prev_labels[r]
        else:
            # This should not happen in your case if all clusters map or split
            raise ValueError("Unexpected unmatched cluster found.")

    # Resolve splits into sub-labels (e.g., 1 -> 1_1, 1_2)
    for prev_label, split_candidates in split_clusters.items():
        for i, split_label in enumerate(split_candidates, 1):
            new_label_mapping[split_label] = f"{prev_label}_{i}"

    # Ensure all current_labels are mapped
    for label in current_labels_unique:
        if label not in new_label_mapping:
            print(f"Warning: Label {label} was not mapped; keeping original label.")
            new_label_mapping[label] = f"new_{label}"  # Assign original label if unmapped

    # Create the new labels array
    try:
        new_labels = np.array([new_label_mapping[label] for label in current_labels])
    except KeyError as e:
        print(f"KeyError: {e} not found in new_label_mapping.")
        print(f"Current Labels: {current_labels}")
        print(f"New Label Mapping: {new_label_mapping}")
        raise
    return new_labels




def relabel_clusters_new(a1, a2):
    # Extract unique labels and their counts
    labels1, counts1 = a1
    labels2, counts2 = a2

    # Create a mapping for the new labels
    new_labels = []
    for idx, label2 in enumerate(labels2):
        # Check if the label exists in the first array and has the same count
        if label2 in labels1:
            index1 = np.where(labels1 == label2)[0][0]
            if counts2[idx] == counts1[index1]:
                new_labels.append(label2)
            else:
                # If counts don't match, it's a split cluster
                parent_label = label2
                # Find how many splits occurred for this parent
                split_count = sum(
                    (counts2[j] < counts1[index1]) for j in range(len(labels2)) if labels2[j] == parent_label)
                new_labels.append(f"{parent_label}_{split_count}")


    return new_labels, counts2



output_path = ut.set_output_path(script_name="combined_pipeline")

run = "alpha_run_ISF"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

df_load = pd.read_csv('../../../Data/Segments/Orion_labeled_segments_KNN_300_15-11-23.csv')
error_sampling_df = pd.read_csv("../../../Data/Gaia/Gaia_DR3_500pc_10percent.csv")

# in case of using segmented data
chunk_labels = df_load.region
chunk = 0
df_chunk = df_load[df_load.region == chunk]

num_sf = 200
# draw number of scale factors
sfs, means = lhc_lloyd('../../../Data/Scale_factors/' + f'sfs_region_{chunk}.txt', num_sf)

# determine means for clusterer initialization
scale_factor_means = {'pos': {'features': ['ra', 'dec', 'parallax'], 'factor': list(means[:3])},
                      'vel': {'features': ['pmra', 'pmdec'], 'factor': list(means[3:])}}

print(scale_factor_means)

# dict for final clustering
dict_final = dict(alpha=-np.inf,
                  beta=0.99,
                  knn_initcluster_graph=35,
                  KNN_list=[20],
                  sfs=sfs,
                  scaling=None,
                  bh_correction=False)

# SigMA kwargs
sigma_kwargs = dict(cluster_features=['ra', 'dec', 'parallax', 'pmra', 'pmdec'], scale_factors=scale_factor_means,
                    nb_resampling=0,
                    max_knn_density=max(dict_final["KNN_list"]) + 1, beta=dict_final["beta"],
                    knn_initcluster_graph=dict_final["knn_initcluster_graph"])

setup_dict = {"scale_factors": scale_factor_means, "sigma_kwargs": sigma_kwargs}

KNNs = dict_final["KNN_list"]
# initialize SigMA with sf_mean
clusterer = SigMA(data=df_chunk, **sigma_kwargs)

labels, pvalues = clusterer.run_sigma(alpha=-np.inf, knn=20, return_pvalues=True)
small_ps = sorted([p for p in pvalues if p < 0.35])

df_w_labels = df_chunk.copy()

consistent_labels = np.empty(shape=(len(small_ps), len(df_chunk)), dtype=object)
# Loop through all p_ids and re-label clusters consistently
for p_id, p in enumerate(small_ps[14:]):
    print(f"-- Current run with p = {p} -- \n")

    # Fit the clusterer and get the labels
    clusterer.fit(alpha=p, knn=20, bh_correction=dict_final["bh_correction"])
    label_array = clusterer.labels_
    labels_real = LabelEncoder().fit_transform(label_array)

    unique_labels = np.unique(labels_real, return_counts=True)
    print(unique_labels)
    # Compute centroids for the current clustering
    current_centroids = compute_centroids(labels_real, df_chunk)

    # For the first clustering result, no need to match, just assign the labels
    if p_id == 0:
        prev_centroids = current_centroids
        consistent_labels[p_id, :] = labels_real

        df_w_labels[f"consistent_labels_{p_id}"] = consistent_labels[p_id, :]
        all_clusters = plot_darkmode(labels=df_w_labels[f"consistent_labels_{p_id}"], df=df_w_labels,
                                     filename=f"consistent_p_{p_id}_{p}", output_pathname=output_path, return_fig=True)

        prev_labels = unique_labels

    else:
        # Match current clusters with the previous ones
        new_labels, unmatched = relabel_clusters_new(prev_labels,unique_labels)
        print(np.unique(new_labels, return_counts=True))

        if np.all(np.unique(new_labels) == np.unique(labels_real)):
            print(f"No changes in run {p_id}")
'''
        else:
            df_w_labels[f"consistent_labels_{p_id}"] = new_labels
            all_clusters = plot_darkmode(labels=df_w_labels[f"consistent_labels_{p_id}"], df=df_w_labels,
                                         filename=f"consistent_p_{p_id}_{p}", output_pathname=output_path,
                                         return_fig=True)

'''