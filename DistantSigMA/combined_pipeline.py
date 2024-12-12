# Python modules
import os
import itertools

import pandas as pd

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

# -------------------- Data ----------------------------

# load the dataframe
df_load = pd.read_csv('../Data/Segments/Orion_labeled_segments_KNN_300_15-11-23.csv')

# load data for error sampling (already slimmed)
error_sampling_df = pd.read_csv("../Data/Gaia/Gaia_DR3_500pc_10percent.csv")

chunks = df_load.region.unique()

analyze_cliques_v = [0.8, 0.9, 0.95]
threshold = [0.2, 0.3, 0.4, 0.5]
threshold_minor = [0.6, 0.7, 0.8, 0.9]
combinations = list(itertools.product(analyze_cliques_v, threshold, threshold_minor))
print(len(combinations))


for r,combo in enumerate(combinations[:]):
    if r >= 0:
        print(combo)
        run = f"combined_pipeline_run_{r}"
        output_path = output_path + f"{run}/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        end_dfs = []

        for chunk_label in df_load.region.unique()[1:]:

            # in case of using segmented data
            chunk = chunk_label
            df_chunk = df_load[df_load.region == chunk]
            print(f"Chunk no: {chunk}. Chunk size: {df_chunk.shape[0]}")

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
            num_sf = 50

            # draw number of scale factors
            sfs, means = lhc_lloyd('../Data/Scale_factors/' + f'sfs_region_{chunk}.txt', num_sf)

            # determine means for clusterer initialization
            scale_factor_means = {'pos': {'features': ['ra', 'dec', 'parallax'], 'factor': list(means[:3])},
                                  'vel': {'features': ['pmra', 'pmdec'], 'factor': list(means[3:])}}

            # dict for final clustering
            dict_final = dict(alpha=0.01,
                              beta=0.99,
                              knn_initcluster_graph=35,
                              KNN_list=[20, 30],
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

            #
            try:
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
                graph = NxGraphAssistant.remove_edges_with_minor(G= cc.G, similarity='weight', threshold=combo[1], threshold_minor=combo[2])

                # FIXME optimize me
                # Merging thresholds for cliques (inner average Jaccard distance)
                graph = NxGraphAssistant.analyze_cliques(graph, combo[0])

                # ----------------- Plotting -------------------
                plotter = PlotHandler(translation, df_final, "")
                labels = clusterMaster.full_pipline_tree_hierachy(
                    graph, Custom_Tree.alex_optimal, translation, plotter, 2, 1)

                df_end = df_chunk.copy()
                df_end["cluster_label"] = labels
                end_dfs.append(df_end)

            except KeyError:
                with open(output_path + 'error_output.txt', 'a') as file:
                    file.write(f"Error with run {r}, chunk {chunk_label}")

                   # df_end = pd.DataFrame()
                   # end_dfs.append(df_end)

            # plotter.plot_labels_3D(labels)
            # plotter.plot_labels_2D(labels)

        # ------------------ Concatenating chunks ------------------------
        # Concatenate datasets and adjust labels
        concatenated_dfs = []
        label_counter = 0

        for df in end_dfs:
            df_with_new_labels = df.copy()
            mask = df['cluster_label'] != -1
            df_with_new_labels.loc[mask, 'cluster_label'] += label_counter
            concatenated_dfs.append(df_with_new_labels)
            label_counter += df['cluster_label'].max() + 1

        # Concatenate all dataframes together
        region_label_df = pd.concat(concatenated_dfs, axis=0, ignore_index=True)

        region_label_df.to_csv(output_path + f"run_{run}_combo_{combo}_final.csv")
        all_clusters = plot(region_label_df["cluster_label"], region_label_df, f"all_clusters", output_pathname=output_path)
        output_path = ut.set_output_path(script_name="combined_pipeline")


    #    r += 1
