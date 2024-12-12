import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture

from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse
from astroML.stats import sigmaG

from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility

import numpy as np
import plotly.graph_objects as go


# Function to plot ellipsoid from covariance matrix and mean
def plot_ellipsoid(mean, cov_matrix, n_points=50):
    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Radii are the square roots of the eigenvalues
    radii = np.sqrt(eigenvalues)

    # Parametric angles
    theta = np.linspace(0, np.pi, n_points)  # Polar angle
    phi = np.linspace(0, 2 * np.pi, n_points)  # Azimuthal angle
    theta, phi = np.meshgrid(theta, phi)

    # Parametric equations for the canonical ellipsoid
    x = radii[0] * np.sin(theta) * np.cos(phi)
    y = radii[1] * np.sin(theta) * np.sin(phi)
    z = radii[2] * np.cos(theta)

    # Combine the canonical coordinates into a matrix
    ellipsoid_points = np.array([x.ravel(), y.ravel(), z.ravel()])

    # Apply the rotation using the eigenvectors
    ellipsoid_rotated = eigenvectors @ ellipsoid_points

    # Reshape back to the original shape
    x_rot = ellipsoid_rotated[0, :].reshape(n_points, n_points) + mean[0]
    y_rot = ellipsoid_rotated[1, :].reshape(n_points, n_points) + mean[1]
    z_rot = ellipsoid_rotated[2, :].reshape(n_points, n_points) + mean[2]

    return x_rot, y_rot, z_rot


sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
script_name = my_utility.get_calling_script_name(__file__)
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/', script_name=script_name)

run = "xdgmm"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

cols = ['source_id', 'ra',
       'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pm',
       'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr',
       'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
       'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
       'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr',
       'astrometric_sigma5d_max', 'ruwe', 'phot_g_mean_flux',
       'phot_g_mean_flux_error', 'phot_g_mean_mag', 'phot_bp_mean_flux',
       'phot_bp_mean_flux_error', 'phot_bp_mean_mag', 'phot_rp_mean_flux',
       'phot_rp_mean_flux_error', 'phot_rp_mean_mag', 'radial_velocity',
       'radial_velocity_error', 'l', 'b', 'fidelity_v2', 'X', 'Y', 'Z',
       'v_a_lsr', 'v_d_lsr', 'U', 'V', 'W', 'mag_abs_g', 'bp_rp', 'g_rp',
       'region', 'cluster_label', 'catalog', 'duplicated_source', 'teff_val',
       'a_g_val', 'e_bp_min_rp_val', 'radius_val', 'lum_val', 'probability',
       'stability']

df_200 = pd.read_csv("../../Data/Comparison/Full_result_table.csv", usecols=cols)
df_200.loc[:, "catalog"] = "SigMA"

# sub data
sub = pd.read_csv("../../Data/Comparison/Ages_new_tuning_v2.csv")

# all_clusters = plot_darkmode(labels=df_200["cluster_label"], df=df_200, filename=f"all_clusters", output_pathname=output_path)

df_clusters = df_200[df_200.cluster_label != -1]  # -1 == field stars
cluster_features = ['X', 'Y', 'Z', 'v_a_lsr', "v_d_lsr"]


# Loop over the clusters
simulated_dfs = []
for group in np.unique(df_clusters["cluster_label"])[:1]:

    # define subset for length check
    subset = df_clusters[df_clusters["cluster_label"] == group]

    X = np.vstack([subset.X.to_numpy(),subset.Y.to_numpy(), subset.Z.to_numpy()]).T

    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([0, 0, 0]).T
    # 2. Fit Extreme Deconvolution with 2 components
    n_components = 2
    xd = XDGMM(n_components, max_iter=200)
    xd.fit(X, Xerr)

    # 3. Extract the fitted parameters (means and covariances of the Gaussian components)
    means = xd.mu
    covariances = xd.V

    X_sample = xd.sample(X.shape[0])

    # 4. Visualize the data and the Gaussian components using Plotly


    # Combine the data and components in one figure
    f = go.Figure()
    # Plot the original data points
    f.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.7),
        name='Data'
    ))

    for i in range(len(means)):
        mean = means[i]
        cov_matrix = covariances[i]

        # Get the transformed ellipsoid points
        x_rot, y_rot, z_rot = plot_ellipsoid(mean, cov_matrix)

        # Add each ellipsoid to the plot
        f.add_trace(go.Surface(x=x_rot, y=y_rot, z=z_rot, colorscale='Viridis', opacity=0.6))

    # Customize layout
    f.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Display the plot
    plot(f)

