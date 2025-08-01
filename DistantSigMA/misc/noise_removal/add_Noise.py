import os
import numpy as np
import pandas as pd

from coordinate_transformations.sky_convert import transform_sphere_to_cartesian, transform_gal_cartesian_and_vtan_to_icrs_pm
from noise_removal_visualization import plot_3D_data
import DistantSigMA.misc.utilities as ut

# Paths
output_path = ut.set_output_path(script_name="DeNoise")

run = "Task_1_addNoise"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)


def add_uniform_Noise(noise_percentage: float, df: pd.DataFrame, label_col: str,
                      pos_range_extend_p: float = 0.2, vel_range_extend: int = 5, parameters: list = None,
                      spherical_parameters:bool = True):
    # --- Setup ---
    if parameters is None:
        # parameters = ['X', 'Y', 'Z', 'v_a_lsr', 'v_d_lsr']
        parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']

    df_noise_all = []

    # Loop over each cluster
    for label, cluster_df in df.groupby(label_col):
        # Step 1: Bounds for X, Y, Z
        pos = cluster_df[parameters[:3]].values
        min_pos= pos.min(axis=0)
        max_pos = pos.max(axis=0)
        range_pos = max_pos - min_pos
        low_pos = min_pos - pos_range_extend_p * range_pos
        high_xyz = max_pos + pos_range_extend_p * range_pos

        if parameters[2] in ["parallax", "plx", "Plx", "Parallax"]:
            low_pos[2] = max(0, low_pos[2])  # do not allow negative parallaxes

        # Step 2: Bounds for velocity1, velocity2
        vel = cluster_df[parameters[3:]].values
        mean_vel = vel.mean(axis=0)
        std_vel = vel.std(axis=0)
        low_vel = mean_vel - vel_range_extend * std_vel
        high_vel = mean_vel + vel_range_extend * std_vel

        # Step 3: Generate noise points
        n_noise = int(noise_percentage * len(cluster_df))  # e.g., 10% noise per cluster
        noise_pos = np.random.uniform(low=low_pos, high=high_xyz, size=(n_noise, 3))
        noise_vel = np.random.uniform(low=low_vel, high=high_vel, size=(n_noise, 2))
        noise_data = np.hstack([noise_pos, noise_vel])

        # Step 4: If input data is spherical, calculate the galactic cartesian values
        if spherical_parameters:
            spherical_df = pd.DataFrame(noise_data, columns=parameters)

            cartesian_df = transform_sphere_to_cartesian(ra=noise_data[:, 0],
                                                         dec=noise_data[:, 1],
                                                         parallax=noise_data[:, 2],
                                                         pmra=noise_data[:, 3],
                                                         pmdec=noise_data[:, 4], )
        # Else, calculate the ICRS values
        else:
            cartesian_df = pd.DataFrame(noise_data, columns=parameters)

            spherical_df = transform_gal_cartesian_and_vtan_to_icrs_pm(X=noise_data[:, 0],
                                                         Y=noise_data[:, 1],
                                                         Z=noise_data[:, 2],
                                                         v_a_lsr=noise_data[:, 3],
                                                         v_d_lsr=noise_data[:, 4], )

        # Step 5: Combine spherical and Cartesian
        df_noise = pd.concat([spherical_df.reset_index(drop=True),
                              cartesian_df.reset_index(drop=True)], axis=1)

        # Step 6: Add label column for noise
        df_noise[label_col] = -1

        # Step 7: Store result
        df_noise_all.append(df_noise)

    # Combine all noise dataframes
    df_noise_all = pd.concat(df_noise_all, ignore_index=True)

    # Append to original DataFrame
    df_augmented = pd.concat([df, df_noise_all], ignore_index=True)

    return df_augmented


if __name__ == "__main__":

    mock_clusters = pd.read_csv("datafiles/Mock_clusters_only_new_cartesian_QC.csv")


    # # # # Sanity Check # # # #
    # print("All rows:", len(data), "\nCluster rows:", len(clusters_only), "\nDifference:",
    #       len(data) - len(clusters_only))
    # plot for quick check
    fig1 = plot_3D_data(data=mock_clusters, labels=mock_clusters.labels, ax_range=[-1000, 1000],
                       xyz_axes=["X", "Y", "Z"])
    fig1.write_html(output_path + f"mock_clusters_cartesian.html")

    fig2 = plot_3D_data(data=mock_clusters, labels=mock_clusters.labels, ax_range=[-300, 300],
                       xyz_axes=["ra", "dec", "parallax"])
    fig2.write_html(output_path + f"mock_clusters_spherical.html")

    # Add new noise
    df_new_noise = add_uniform_Noise(noise_percentage=.9,
                                     df=mock_clusters,
                                     label_col="labels",
                                     #parameters=["X", "Y", "Z", "v_a_lsr", "v_d_lsr"]
                                     )

    fig_icrs = plot_3D_data(data=df_new_noise, labels=df_new_noise.labels,
                            xyz_axes=["ra", "dec", "parallax"],
                            ax_range=[-300, 300]
                            )

    fig_icrs.write_html(output_path + "Mock_clusters_new_noise_icrs.html")

    fig_cart = plot_3D_data(data=df_new_noise, labels=df_new_noise.labels,

                            ax_range=[-1000, 1000]
                            )

    fig_cart.write_html(output_path + "Mock_clusters_new_noise_cartesian.html")

    df_new_noise.to_csv("datafiles/Mock_clusters_newNoise.csv")


