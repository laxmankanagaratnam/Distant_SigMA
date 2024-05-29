import sys
import os
import pandas as pd
import numpy as np
from DistantSigMA.Analysis.IsochroneArchive.myTools import my_utility
from MockData.cluster_simulations import SimulateCluster

# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/')

output_path += "Simulations/"
run = 1

if not os.path.exists(output_path + f"Run_{run}/"):
    os.makedirs(output_path + f"Run_{run}/")

# read in sampling data (already slimmed)
error_sampling_df = pd.read_csv("/Data/Gaia_DR3_500pc_10percent.csv")

# features to calculated distribution for
cluster_features = ['X', 'Y', 'Z', 'v_a_lsr', "v_d_lsr"]

# clean selection of Region 0
df_load = pd.read_csv("../../Data/Region_0/Region_0_run6_tc_cleaned.csv")

# 6C
df_load.loc[df_load['non_C4'], 'rsc'] = -1

# 4C
# df_load.loc[df_load['non_C3'], 'rsc'] = -1

df_load.loc[df_load['non_C1'], 'rsc'] = -1
df_load.loc[df_load['C3_split'], 'rsc'] = -1

df_clusters = df_load[df_load.rsc != -1]

stds = np.empty(shape=(5, len(np.unique(df_clusters["rsc"]))))

e_convolved_dfs = []
for group in np.unique(df_clusters["rsc"])[:]:
    sim = SimulateCluster(region_data=df_clusters, group_id=group, clustering_features=cluster_features)
    # print(f"Group_{group}: {sim.data.shape[0]} stars / {sim.rv.shape[0]} RVs, mean(RV) = {round(sim.rv.mean(),2)},"
    #      f" std(RV) = {round(sim.rv.std(),2)}")

    print(f"Clean Group_{group}: {sim.data.shape[0]} stars / {sim.cleaned_rv['radial_velocity'].shape[0]} RVs, "
          f"mean(RV) = {round(sim.cleaned_rv['radial_velocity'].mean(), 2)},"
          f" std(RV) = {round(sim.cleaned_rv['radial_velocity'].std(), 2)}")

'''    
    
    e_convolved_cluster = sim.error_convolve(sampling_data=error_sampling_df)

    sim_df = pd.DataFrame(data=sim.e_convolved_points, columns=["ra", "dec", "plx", "pmra", "pmdec", "X", "Y", "Z"]) \
        .assign(label=int(group))
    e_convolved_dfs.append(sim_df)

    # std cols
    std_columns = ["ra", "dec", "plx", "pmra", "pmdec"]
    stds[:, group] = sim_df[std_columns].std().values
    # print(f"{group}: {sim_df[std_columns].std().to_numpy()}")

    # resampled data histograms
    fig = sim.diff_histogram(e_convolved_cluster)
    fig.show()
    plt.savefig(output_path + f"Run_{run}/"+f"Group_{sim.group_id}.pdf", dpi=300)

convolved_df = pd.concat(e_convolved_dfs, ignore_index=True)
convolved_df.to_csv(output_path + f"Run_{run}/" + "Simulated_clusters_labeled_Region0_run10.csv")

for i, label in enumerate(["ra", "dec", "plx", "pmra", "pmdec"]):
    print(f"{label}:", np.min(stds[i, :]), np.mean(stds[i, :]), np.max(stds[i, :]))

# outer histogram of the group stds
outer_fig, ax = plt.subplots(2, 3, figsize=(7, 4))
ax = ax.ravel()

for i, label in enumerate(["ra", "dec", "parallax", "pmra", "pmdec"]):
    data_column = stds[i, :]
    num_bins_data = sim.Knuths_rule(data_column)
    ax[i].hist(data_column, bins=len(data_column), facecolor="green", edgecolor='black')
    ax[i].set_title(f"{label} ({round(min(data_column),3)}, {round(max(data_column),3)})")


plt.suptitle(f"Outer hist")
plt.tight_layout()
plt.show()

outer_fig.savefig(output_path + f"Run_{run}/"+"outer_distributions.png", dpi=300)
'''
