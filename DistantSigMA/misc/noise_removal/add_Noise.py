import os
import numpy as np
import pandas as pd

from noise_removal_visualization import plot_3D_data
import DistantSigMA.misc.utilities as ut

# Paths
output_path = ut.set_output_path(script_name="DeNoise")

run = "Task_1_Add_noise"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)


def add_uniform_Noise(noise_percentage: float, df: pd.DataFrame, label_col: str,
                      pos_range_extend_p: float = 0.2, vel_range_extend: int = 5, parameters: list = None):
    # --- Setup ---
    if parameters is None:
        # parameters = ['X', 'Y', 'Z', 'v_a_lsr', 'v_d_lsr']
        parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']

    df_noise_all = []

    # Loop over each cluster
    for label, cluster_df in df.groupby(label_col):
        # Step 1: Bounds for X, Y, Z
        xyz = cluster_df[parameters[:3]].values
        min_xyz = xyz.min(axis=0)
        max_xyz = xyz.max(axis=0)
        range_xyz = max_xyz - min_xyz
        low_xyz = min_xyz - pos_range_extend_p * range_xyz
        high_xyz = max_xyz + pos_range_extend_p * range_xyz

        # Step 2: Bounds for velocity1, velocity2
        vel = cluster_df[parameters[3:]].values
        mean_vel = vel.mean(axis=0)
        std_vel = vel.std(axis=0)
        low_vel = mean_vel - vel_range_extend * std_vel
        high_vel = mean_vel + vel_range_extend * std_vel

        # Step 3: Generate noise points
        n_noise = int(noise_percentage * len(cluster_df))  # e.g., 10% noise per cluster
        noise_xyz = np.random.uniform(low=low_xyz, high=high_xyz, size=(n_noise, 3))
        noise_vel = np.random.uniform(low=low_vel, high=high_vel, size=(n_noise, 2))
        noise_data = np.hstack([noise_xyz, noise_vel])

        # Step 4: Create noise DataFrame
        df_noise = pd.DataFrame(noise_data, columns=parameters)
        df_noise[label_col] = -1

        df_noise_all.append(df_noise)

    # Combine all noise dataframes
    df_noise_all = pd.concat(df_noise_all, ignore_index=True)

    # Append to original DataFrame
    df_augmented = pd.concat([df, df_noise_all], ignore_index=True)

    return df_augmented


if __name__ == "__main__":
    mock_df = pd.read_csv("/Users/alena/PycharmProjects/Distant_SigMA/Data/Mock/Box_0_with_all_clustering_results.csv")

    # Quality cuts
    data = mock_df.loc[mock_df['parallax'] > 0]
    data = data.loc[data['parallax'] / data['parallax_error'] > 4.5].reset_index(drop=True)

    # remove Noise
    clusters_only = data[data.labels != -1]

    # # # # Sanity Check # # # #
    # print("All rows:", len(data), "\nCluster rows:", len(clusters_only), "\nDifference:",
    #       len(data) - len(clusters_only))
    # plot for quick check
    # fig = plot_3D_data(data=data, labels=data.labels, ax_range=[-1000, 1000])
    # fig.write_html(output_path + f"all_mock_clusters.html")
    # fig = plot_3D_data(data=clusters_only, labels=clusters_only.labels, ax_range=[-1000, 1000])
    # fig.write_html(output_path + f"all_mock_clusters_no_noise.html")

    # Add new noise
    df_new_noise = add_uniform_Noise(noise_percentage=.9,
                                     df=clusters_only,
                                     label_col="labels",
                                     #parameters=["X", "Y", "Z", "v_a_lsr", "v_d_lsr"]
                                     )

    fig = plot_3D_data(data=df_new_noise, labels=df_new_noise.labels,
                       xyz_axes=["ra", "dec", "parallax"],
                       ax_range=[-300, 300]
                       )

    fig.write_html(output_path+"Mock_clusters_new_noise_ICRS.html")

    df_new_noise.to_csv("datafiles/Mock_clusters_newNoise.csv")