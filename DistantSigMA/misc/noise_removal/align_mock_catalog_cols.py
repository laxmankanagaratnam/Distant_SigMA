import os
import pandas as pd

from coordinate_transformations.sky_convert import transform_sphere_to_cartesian, transform_gal_cartesian_and_vtan_to_icrs_pm
from noise_removal_visualization import plot_3D_data
import DistantSigMA.misc.utilities as ut

""" 
    The ICRS coordinates and galactic X,Y,Z coordinates in the input mock catalog are not naturally transformable
    with standard transformation functions. Here, I create two new input catalogs that have
    1. Original ICRS + newly transformed XYZ coordinates
    2. Original XYZ + newly transformed ICRS coordinates
"""

# Paths
output_path = ut.set_output_path(script_name="DeNoise")

run = "Mock_catalog_ICRS_gal_align"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load input catalog and do some quality filtering
mock_df = pd.read_csv("/Users/alena/PycharmProjects/Distant_SigMA/Data/Mock/Box_0_with_all_clustering_results.csv")

# Quality cuts
data = mock_df.loc[mock_df['parallax'] > 0]
data = data.loc[data['parallax'] / data['parallax_error'] > 4.5].reset_index(drop=True)

# Throw away the noise (it is wrong)
clusters_only = data[data.labels != -1]
clusters_only =clusters_only.reset_index(drop=True)

# Sanity check
# ------------------------------------------

drop = "spherical"

# 1: Drop cartesian coordinates
if drop == "cartesian":

    df1 = clusters_only.drop(columns=["X", "Y", "Z", "v_a_lsr", "v_d_lsr"])

    # Recalculate cartesian coordinates
    new_cart = transform_sphere_to_cartesian(ra=df1.ra.to_numpy(),
                                             dec=df1.dec.to_numpy(),
                                             parallax=df1.parallax.to_numpy(),
                                             pmra=df1.pmra.to_numpy(),
                                             pmdec=df1.pmdec.to_numpy(),
                                             )

    df1_all = pd.concat([df1, new_cart], axis=1)
    print(df1.shape, new_cart.shape, df1_all.shape)
    df1_all.to_csv("datafiles/Mock_clusters_only_new_cartesian_QC.csv")

    # plot two figures
    fig_cart_new = plot_3D_data(data=df1_all, labels=df1_all.labels,
                            xyz_axes=["X", "Y", "Z"],
                            ax_range=[-1000, 1000]
                            )
    fig_cart_new.write_html(output_path + "Mock_clusters_only_new_cartesian.html")

    fig_icrs = plot_3D_data(data=df1_all, labels=df1_all.labels,
                            xyz_axes=["ra", "dec", "parallax"],
                            ax_range=[-300, 300]
                            )
    fig_icrs.write_html(output_path + "Mock_clusters_only_original_spherical.html")

# 2: Drop ICRS coordinates
elif drop == "spherical":

    df2 = clusters_only.drop(columns=["ra", "dec", "parallax", "pmra", "pmdec"])

    new_icrs = transform_gal_cartesian_and_vtan_to_icrs_pm(X=df2.X.to_numpy(),
                                                           Y=df2.Y.to_numpy(),
                                                           Z=df2.Z.to_numpy(),
                                                           v_a_lsr=df2.v_a_lsr.to_numpy(),
                                                           v_d_lsr=df2.v_d_lsr.to_numpy(),
                                                           )
    df2_all = pd.concat([df2, new_icrs], axis=1)
    print(df2.shape, new_icrs.shape, df2_all.shape)
    df2_all.to_csv("datafiles/Mock_clusters_only_new_spherical_QC.csv")


    # plot two figures
    fig_cart = plot_3D_data(data=df2_all, labels=df2_all.labels,
                            xyz_axes=["X", "Y", "Z"],
                            ax_range=[-1000, 1000]
                            )

    fig_cart.write_html(output_path + "Mock_clusters_only_original_cartesian.html")

    fig_icrs_new = plot_3D_data(data=df2_all, labels=df2_all.labels,
                            xyz_axes=["ra", "dec", "parallax"],
                            ax_range=[-300, 300]
                            )

    fig_icrs_new.write_html(output_path + "Mock_clusters_only_new_spherical.html")

