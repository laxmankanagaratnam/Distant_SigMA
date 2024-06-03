# Python modules
import os

#### DistantSigMA modules
from misc import utilities as ut
from DistantSigMA.clustering_routine import *
from DistantSigMA.DistantSigMA.cluster_simulations import calculate_std_devs
from DistantSigMA.DistantSigMA.scalefactor_sampling import lhc_lloyd

# Tree/Graph modules
from alex_workspace.ClusterHandler import ClusteringHandler
from alex_workspace.NxGraphAssistant import NxGraphAssistant
from alex_workspace.PlotHandler import PlotHandler
from alex_workspace.Tree import Custom_Tree

# -------------------- Setup ----------------------------

# Paths
output_path = ut.set_output_path(script_name="combined_pipeline")

run = "test_combined_pipeline"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# -------------------- Data ----------------------------

# load the dataframe
df_load = pd.read_csv('../Data/Segments/Orion_labeled_segments_KNN_300_15-11-23.csv')

# load data for error sampling (already slimmed)
error_sampling_df = pd.read_csv("../Data/Gaia/Gaia_DR3_500pc_10percent.csv")

# in case of using segmented data
chunk_labels = df_load.region
chunk = 0
df_chunk = df_load[df_load.region == chunk]

# extend output path
result_path = output_path + f"Region_{int(chunk)}/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# =================== Clustering ========================

# ------------ A) Preliminary solution -------------------

# 1.) Parameters
dict_prelim = dict(alpha=0.01,
                   beta=0.99,
                   knn_initcluster_graph=35,
                   KNN_list=[15, 20, 25, 30],
                   sfs=[0.1, 0.15, 0.2, 0.25, 0.3,
                        0.35, 0.4, 0.45, 0.5, 0.55],
                   scaling="robust",
                   bh_correction=True)

print(f"PART A) Starting clustering ... \n")

df_prelim = run_clustering(region_label=chunk, df_input=df_chunk,
                           sf_params="parallax_scaled",
                           parameter_dict=dict_prelim, mode="prelim",
                           output_loc=result_path)

# --------------- B) Simulate clusters -------------------

print(f"PART B) Simulating clusters ... \n")

stds = calculate_std_devs(input_df=df_prelim, SigMA_dict=dict_prelim,
                          sampling_data=error_sampling_df, n_artificial=1,
                          output_path=result_path, plot_figs=False)

# save scaling factors in Data directory
directory = "/Users/alena/PycharmProjects/Distant_SigMA/Data/Scale_factors"
filename = f"sfs_region_{chunk}.txt"

# Open the file in write mode ('w')
with open(f"{directory}/{filename}", 'w') as file:
    for i, label in enumerate(["ra", "dec", "parallax", "pmra", "pmdec"]):
        print(f"{label}:", np.min(stds[i, :]), np.mean(stds[i, :]), np.max(stds[i, :]),
              file=file)

# ------------------ C) Cluster with new SF ------------------------

print(f"PART C) Re-evaluating clustering ... \n")

# determine the number of SF to draw using lhc_lloyd sampling
num_sf = 150

# draw number of scale factors
sfs, means = lhc_lloyd('../Data/Scale_factors/' + f'sfs_region_{chunk}.txt', num_sf)

# determine means for clusterer initialization
scale_factor_means = {'pos': {'features': ['ra', 'dec', 'parallax'], 'factor': list(means[:3])},
                      'vel': {'features': ['pmra', 'pmdec'], 'factor': list(means[3:])}}

# dict for final clustering
dict_final = dict(alpha=0.01,
                  beta=0.99,
                  knn_initcluster_graph=35,
                  KNN_list=[30],
                  sfs=sfs,
                  scaling=None,
                  bh_correction=False)

# Generate grouped solutions
grouped_labels = partial_clustering(df_input=df_chunk,
                                    sf_params=["ra", "dec", "parallax", "pmra", "pmdec"],
                                    parameter_dict=dict_final, mode="final",
                                    output_loc=result_path,
                                    column_means=scale_factor_means)

print(grouped_labels.shape)
# add the grouped solution labels to the dataframe holding the observations
df_final = df_chunk.copy()
for col in range(grouped_labels.shape[0]):
    df_final.loc[:, f"cluster_label_group_{col}"] = grouped_labels[col, :]

# df_final.to_csv(result_path+f"Region_{chunk}_sf_{num_sf}_grouped_solutions.csv")

# ----------------- D) Graph/Tree application -------------------

# Creating instances of custom classes
clusterMaster = ClusteringHandler()  # Manages cluster operations

# make consensus over the grouped labels that came out of step C)
cc = ClusterConsensus(*grouped_labels)  # creates graph object cc.G
translation = cc.labels_bool_dict2arr

# rename the similarity in every edge to "weight"
for i, j, d in cc.G.edges(data=True):
    d['weight'] = d['similarity']
    d['weight_minor'] = d['similarity_minor']
    del d['similarity']
    del d['similarity_minor']

# FIXME optimize me
# Jaccard (minor) distances
graph = NxGraphAssistant.remove_edges_with_minor(cc.G, 'weight', 0.3, 0.7)

# FIXME optimize me
# Merging thresholds for cliques (inner average Jaccard distance)
graph = NxGraphAssistant.analyze_cliques(graph, 0.95)

# ----------------- Plotting -------------------
plotter = PlotHandler(translation, df_final, "")
labels = clusterMaster.full_pipline_tree_hierachy(
    graph, Custom_Tree.alex_optimal, translation, plotter, 2, 1)

plotter.plot_labels_3D(labels)
plotter.plot_labels_2D(labels)
