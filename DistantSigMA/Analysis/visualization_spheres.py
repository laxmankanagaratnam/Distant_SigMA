import sys
import os
import numpy as np
import pandas as pd

from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility
from DistantSigMA.DistantSigMA.PlotlyResults import plot_darkmode, plot_surface
from DistantSigMA.DistantSigMA.cluster_simulations import SimulateCluster


# Paths
# ---------------------------------------------------------
# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
script_name = my_utility.get_calling_script_name(__file__)
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/', script_name=script_name)

run = "spheres"
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
for group in np.unique(df_clusters["cluster_label"])[:]:

    # define subset for length check
    subset = df_clusters[df_clusters["cluster_label"] == group]

    subset["age"] = sub.loc[int(group), "age"]
    subset["av"] = sub.loc[int(group), "av"]
    subset["descriptor"] = sub.loc[int(group), "descriptor"]

    sim = SimulateCluster(region_data=df_clusters, group_id=group, clustering_features=cluster_features,
                          label_column="cluster_label", multiplier_fraction=1)
    sim_df = pd.DataFrame(data=sim.simulated_points,
                          columns=["X", "Y", "Z", "v_a_lsr", "v_d_lsr"]).assign(cluster_label=int(group))
    sim_df[["mag_abs_g", "bp_rp", "g_rp", "age", "av"]] = subset.loc[:, ["mag_abs_g", "bp_rp", "g_rp", "age", "av"]].values
    if not "**" in subset["descriptor"].unique()[0]:
        #simulated_dfs.append(sim_df)
        simulated_dfs.append(subset)

    else:
        print(group)
    #simulated_dfs.append(subset)


simulated_points_df = pd.concat(simulated_dfs, ignore_index=True)

fig, x,y = plot_surface(df=simulated_points_df, filename=f"real_{run}_age_rdbu", sub_data=sub, cluster_descriptor="descriptor",
                             icrs=False, return_fig=True)

name = 'eye = (x:1.5, y:2, z:0.1)'
camera = dict(
    eye=dict(x=1., y=.35, z=2.5)
)

fig.update_layout(scene_camera=camera)

#fig.write_html(output_path + f"test_6_position.html")
fig.write_image(output_path+"Elongated_grid3.png", width=500, height=800, scale=3)
