
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Tree import Custom_Tree
from Tree import Custom_tree_node


def plot3D(labels: np.array, df: pd.DataFrame, filename: str, output_pathname: str = None, hrd: bool = False,
         icrs: bool = False, return_fig: bool = False):
    """ Simple function for creating a result plot of all the final clusters. HRD option available."""

    # not relevant for the end result
    if icrs:
        vel1 = "pmra"
        vel2 = "pmdec"
    else:
        vel1 = "v_a_lsr"
        vel2 = "v_d_lsr"

    cs = labels  # set label variable
    df_plot = df.loc[
        cs != -1]  # remove field stars from dataframe (field stars are assigned -1 in SigMA during clustering)
    clustering_solution = cs.astype(int)  # set label array to integers
    clustering_solution = clustering_solution[clustering_solution != -1]  # also remove field stars from label array
    cut_us = np.random.uniform(0, 1, size=clustering_solution.shape[0]) < 0.1  # background cut

    # plotting specs
    bg_opacity = 0.1
    bg_color = 'gray'
    plt_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF',
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
        x=df_plot.loc[cut_us, 'X'], y=df_plot.loc[cut_us, 'Y'], z=df_plot.loc[cut_us, 'Z'],
        mode='markers', marker=dict(size=1, color=bg_color, opacity=bg_opacity), hoverinfo='none', showlegend=False, )
    fig.add_trace(trace_3d_bg, row=1, col=1)

    # sun marker
    trace_sun = go.Scatter3d(
        x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
        mode='markers', marker=dict(size=5, color='red', symbol='x'), hoverinfo='none', showlegend=True, name='Sun')
    fig.add_trace(trace_sun, row=1, col=1)

    # 3D clusters
    for j, uid in enumerate(np.unique(clustering_solution)):  # for each cluster label in the label array
        if uid != -1:
            plot_points = (clustering_solution == uid)  # grab the right locations
            trace_3d = go.Scatter3d(
                x=df_plot.loc[plot_points, 'X'], y=df_plot.loc[plot_points, 'Y'], z=df_plot.loc[plot_points, 'Z'],
                mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)]), hoverinfo='none',
                showlegend=True, name=f'Cluster {int(uid)} ({np.sum(plot_points)} stars)', legendgroup=f'group-{uid}', )
            fig.add_trace(trace_3d, row=1, col=1)  # add cluster trace

    # --------------- 2D vel plot -------------------

    # background
    trace_vel_bg = go.Scatter(
        x=df_plot.loc[cut_us, vel1], y=df_plot.loc[cut_us, vel2],
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
    plt_kwargs = dict(showbackground=False, showline=False, zeroline=True, zerolinecolor='grey', zerolinewidth=2,
                      showgrid=True, showticklabels=True, color="black",
                      linecolor='black', linewidth=1, gridcolor='rgba(100,100,100,0.5)')

    xaxis = dict(**plt_kwargs, title='X [pc]')  # , tickmode = 'linear', dtick = 50, range=[-50,200])
    yaxis = dict(**plt_kwargs, title='Y [pc]')  # , tickmode = 'linear', dtick = 50, range=[-200, 50])
    zaxis = dict(**plt_kwargs, title='Z [pc]')  # , tickmode = 'linear', dtick = 50, range=[-100, 150])

    if not icrs:
        fig.update_xaxes(title_text="v_alpha", showgrid=False, row=1, col=2, color="black")
        fig.update_yaxes(title_text="v_delta", showgrid=False, row=1, col=2, color="black")
    else:
        fig.update_xaxes(title_text="pmra", showgrid=False, row=1, col=2, color="black")
        fig.update_yaxes(title_text="pmdec", showgrid=False, row=1, col=2, color="black")

    # Finalize layout
    fig.update_layout(
        title="",
        # width=800,
        # height=800,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(itemsizing='constant'),
        # 3D plot
        scene=dict(
            xaxis=dict(xaxis),
            yaxis=dict(yaxis),
            zaxis=dict(zaxis)
        )
    )

    # # --------------- HRD plot -------------------
    if hrd:

        # background
        trace_hrd_bg = go.Scatter(x=df_plot.loc[cut_us, 'g_rp'], y=df_plot.loc[cut_us, 'mag_abs_g'], mode='markers',
                                  marker=dict(size=3, color=bg_color, opacity=bg_opacity), hoverinfo='none',
                                  showlegend=False)
        fig.add_trace(trace_hrd_bg, row=1, col=3)

        # HRD of each cluster
        for j, kid in enumerate(np.unique(clustering_solution)):
            if kid != -1:
                plot_points = (clustering_solution == kid)  # & cut_us
                trace_hrd = go.Scatter(x=df_plot.loc[plot_points, 'g_rp'], y=df_plot.loc[plot_points, 'mag_abs_g'],
                                       mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)], ),
                                       hoverinfo='none', legendgroup=f'group-{kid}',
                                       name=f'Cluster {kid} ({np.sum(plot_points)} stars)', showlegend=False)
                fig.add_trace(trace_hrd, row=1, col=3)

        fig.update_xaxes(title_text="G-RP", showgrid=False, row=1, col=3)
        fig.update_yaxes(title_text="Abs mag G", showgrid=False, autorange="reversed", row=1, col=3)

        # Finalize layout
        fig.update_layout(
            title="",
            # width=800,
            # height=800,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(itemsizing='constant'), )

    if output_pathname:
        fig.write_html(output_pathname + f"{filename}.html")

    if return_fig:
        return fig
    
    
    
def plot2D(labels: np.array, df: pd.DataFrame, filename: str, output_pathname: str = None, hrd: bool = False,
        icrs: bool = False, return_fig: bool = False):
        """ Simple function for creating a result plot of all the final clusters. HRD option available."""

        # not relevant for the end result
        if icrs:
            vel1 = "pmra"
            vel2 = "pmdec"
        else:
            vel1 = "v_a_lsr"
            vel2 = "v_d_lsr"

        cs = labels  # set label variable
        df_plot = df.loc[
            cs != -1]  # remove field stars from dataframe (field stars are assigned -1 in SigMA during clustering)
        clustering_solution = cs.astype(int)  # set label array to integers
        clustering_solution = clustering_solution[clustering_solution != -1]  # also remove field stars from label array
        cut_us = np.random.uniform(0, 1, size=clustering_solution.shape[0]) < 0.1  # background cut

        # plotting specs
        bg_opacity = 0.1
        bg_color = 'gray'
        plt_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF',
                    '#FECB52', '#B82E2E', '#316395']

        #  ---------------  Create figure  ---------------
        #  --------------- ---------------  --------------
        # without HRD subplot
 
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
            column_widths=[0.33, 0.33, 0.33],
            subplot_titles=['X vs Y', 'X vs Z', 'Y vs Z'], )



        # --------------- 2D scatter plots -------------------
        # X vs Y
        trace_xy_bg = go.Scatter(
            x=df_plot.loc[cut_us, 'X'], y=df_plot.loc[cut_us, 'Y'],
            mode='markers', marker=dict(size=1, color=bg_color, opacity=bg_opacity), hoverinfo='none', showlegend=False)
        fig.add_trace(trace_xy_bg, row=1, col=1)

        for j, uid in enumerate(np.unique(clustering_solution)):  # for each cluster label in the label array
            if uid != -1:
                plot_points = (clustering_solution == uid)  # grab the right locations
                trace_xy = go.Scatter(
                    x=df_plot.loc[plot_points, 'X'], y=df_plot.loc[plot_points, 'Y'],
                    mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)]), hoverinfo='none',
                    showlegend=True, name=f'Cluster {int(uid)} ({np.sum(plot_points)} stars)', legendgroup=f'group-{uid}', )
                fig.add_trace(trace_xy, row=1, col=1)  # add cluster trace

        # X vs Z
        trace_xz_bg = go.Scatter(
            x=df_plot.loc[cut_us, 'X'], y=df_plot.loc[cut_us, 'Z'],
            mode='markers', marker=dict(size=1, color=bg_color, opacity=bg_opacity), hoverinfo='none', showlegend=False)
        fig.add_trace(trace_xz_bg, row=1, col=2)

        for j, uid in enumerate(np.unique(clustering_solution)):  # for each cluster label in the label array
            if uid != -1:
                plot_points = (clustering_solution == uid)  # grab the right locations
                trace_xz = go.Scatter(
                    x=df_plot.loc[plot_points, 'X'], y=df_plot.loc[plot_points, 'Z'],
                    mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)]), hoverinfo='none',
                    showlegend=True, name=f'Cluster {int(uid)} ({np.sum(plot_points)} stars)', legendgroup=f'group-{uid}', )
                fig.add_trace(trace_xz, row=1, col=2)  # add cluster trace

        # Y vs Z
        trace_yz_bg = go.Scatter(
            x=df_plot.loc[cut_us, 'Y'], y=df_plot.loc[cut_us, 'Z'],
            mode='markers', marker=dict(size=1, color=bg_color, opacity=bg_opacity), hoverinfo='none', showlegend=False)
        fig.add_trace(trace_yz_bg, row=1, col=3)

        for j, uid in enumerate(np.unique(clustering_solution)):  # for each cluster label in the label array
            if uid != -1:
                plot_points = (clustering_solution == uid)  # grab the right locations
                trace_yz = go.Scatter(
                    x=df_plot.loc[plot_points, 'Y'], y=df_plot.loc[plot_points, 'Z'],
                    mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)]), hoverinfo='none',
                    showlegend=True, name=f'Cluster {int(uid)} ({np.sum(plot_points)} stars)', legendgroup=f'group-{uid}', )
                fig.add_trace(trace_yz, row=1, col=3)  # add cluster trace

        # --------------- 2D vel plot -------------------

        # background
        trace_vel_bg = go.Scatter(
            x=df_plot.loc[cut_us, vel1], y=df_plot.loc[cut_us, vel2],
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


        # Finalize layout
        fig.update_layout(
            title="",
            # width=800,
            # height=800,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(itemsizing='constant'), )

        if output_pathname:
            fig.write_html(output_pathname + f"{filename}.html")

        if return_fig:
            return fig
    
    
    
class PlotHandler:
    """
    A class to manage data processing and visualization for different clusters within a given region.

    Attributes:
        output_path (str): Directory path where plots will be saved.
        data_path (str): Directory path where data files are stored.
        df_region (DataFrame): Pandas DataFrame containing region-specific data.
        prefix (str): Prefix used to identify specific columns in data frames.
        count (int): Count of columns that start with the specified prefix.
        tmp (list): Temporary storage used for data manipulation.
        r (int): Index specifying a particular region.
    """
    
    def __init__(self, tmp2, orion_index=0, path="", data_path=r'C:\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\SigMA_Alex_modifications\alex_workspace\3D_plotting\3D_plotting\Region_dataframes/'):
        """
        Initializes the PlotHandler with paths and region-specific data.

        Args:
            tmp2 (list): A temporary list used to store label data.
            orion_index (int): Index for selecting a specific Orion region. Defaults to 0.
            path (str): Output path for saving plots. Defaults to an empty string.
            data_path (str): Path where region data files are located. Defaults to a specified path.
        """
        self.output_path = path
        self.data_path = data_path
        self.tmp = tmp2
        regions = [f'Region_{float(i)}_sf_200_grouped_solutions.csv' for i in range(5)]
        self.r = orion_index
        region = regions[self.r]

        if orion_index == 2:  # Special case for region 2 which is split into two parts.
            region_part_1 = pd.read_csv(data_path + f'Region_{float(2)}_sf_200_grouped_solutions-1.csv')
            region_part_2 = pd.read_csv(data_path + f'Region_{float(2)}_sf_200_grouped_solutions-2.csv')
            self.df_region = pd.concat([region_part_1, region_part_2])
        else:
            self.df_region = pd.read_csv(data_path + region)

        self.prefix = 'cluster_label_group'
        self.count = sum(1 for col in self.df_region.columns if col.startswith(self.prefix))
    
    def labels_single_node(self,node):
        """
        Extracts and processes label data for a single node based on the temporary data list.

        Args:
            node (Custom_tree_node): A node from which labels are extracted.

        Returns:
            numpy.ndarray: An array of label data processed from the node.
        """
        local_list = []
        for sub_node in node.name.split("+"):
            local_list.append(self.tmp[int(sub_node)])
        merged_list = [any(item) for item in zip(*local_list)]
        
        upgraded_list = []
        for i in range(len(merged_list)):
            if merged_list[i] == 1:
                upgraded_list.append(1)
            else:
                upgraded_list.append(-1)
        upgraded_list = np.array(upgraded_list, dtype=float)
        return upgraded_list
        
    def compare_two_clusters(self,list1,list2):
        """
        Compares labels between two clusters and identifies unique and shared labels.

        Args:
            list1 (list): First cluster of nodes.
            list2 (list): Second cluster of nodes.

        Returns:
            numpy.ndarray: An array showing the result of the comparison between the two clusters.
        """
        global_list = []
        local_list = []
        local_list2 = []
        for node in list1:
            local_list.append(self.tmp[node])
        merged_list = [any(item) for item in zip(*local_list)]
        global_list.append(merged_list)
        for node in list2:
            local_list2.append(self.tmp[node])
        merged_list = [any(item) for item in zip(*local_list2)]
        global_list.append(merged_list)
            
        upgraded_list = []
        for i in range(len(global_list[0])):
            for j in range(len(global_list)):
                if global_list[j][i] == 1:
                    upgraded_list.append(j)
                    break
            else:
                upgraded_list.append(-1)
                
        upgraded_list = np.array(upgraded_list, dtype=float)
        return upgraded_list
    
    def compare_two_clusters_marke_shared(self,list1,list2):
        """
        Marks shared labels between two clusters with cluster 2 (list1 = 0, list2 = 1, shared = 2)
        In each of the list all stars are included which are in least one of the clusters 

        Args:
            list1 (list): First cluster of nodes.
            list2 (list): Second cluster of nodes.

        Returns:
            numpy.ndarray: An array marking shared labels between the two clusters.
        """
        global_list = []
        local_list = []
        local_list2 = []
        for node in list1:
            local_list.append(self.tmp[node])
        merged_list = [any(item) for item in zip(*local_list)]
        global_list.append(merged_list)
        for node in list2:
            local_list2.append(self.tmp[node])
        merged_list = [any(item) for item in zip(*local_list2)]
        global_list.append(merged_list)
            
        upgraded_list = []
        for i in range(len(global_list[0])):
            dict = {}
            for j in range(len(global_list)):
                if global_list[j][i] == 1:
                    dict[j] = dict.get(j, 0) + 1
            if len(dict) == 0:
                upgraded_list.append(-1)
            if len(dict) == 1:
                #add only key to list
                upgraded_list.append(list(dict.keys())[0])
            if len(dict) > 1:
                upgraded_list.append(2)
                
        upgraded_list = np.array(upgraded_list, dtype=float)
        return upgraded_list

    def compare_n_clusters(self,list):
        global_list = []
        for list_inner in list:
            local_list = []
            for node in list_inner:
                local_list.append(self.tmp[node])
            merged_list = [any(item) for item in zip(*local_list)]
            global_list.append(merged_list)
            
        upgraded_list = []
        for i in range(len(global_list[0])):
            for j in range(len(global_list)):
                if global_list[j][i] == 1:
                    upgraded_list.append(j)
                    break
            else:
                upgraded_list.append(-1)
                
        upgraded_list = np.array(upgraded_list, dtype=float)
        return upgraded_list

    def print_trees_leaf_low_or(self,tree_list):
        """
        Analyzes and prints labels for the leaves of the trees using logical OR.

        Args:
            tree_list (list): A list of trees whose leaves will be analyzed.

        Returns:
            numpy.ndarray: An array of labels processed using logical OR among the leaves.
        """
        list_alex = []
        # check if tree_list is a list
        if not isinstance(tree_list, list):
            tree_list = [tree_list]
        for tree in tree_list:
            local_list = []
            for node_up in tree.get_leaf_nodes():
                if not isinstance(node_up.name, str):
                    local_list.append(self.tmp[node_up.name])
                    merged_list = [any(item) for item in zip(*local_list)]
                    list_alex.append(merged_list)
                    continue
                for node_down in node_up.name.split("+"):
                    local_list.append(self.tmp[int(node_down)])
                merged_list = [any(item) for item in zip(*local_list)]
                list_alex.append(merged_list)
        upgraded_list = []
        for i in range(len(list_alex[0])):
            for j in range(len(list_alex)):
                if list_alex[j][i] == 1:
                    upgraded_list.append(j)
                    break
            else:
                upgraded_list.append(-1)

        upgraded_list = np.array(upgraded_list, dtype=float)
        return upgraded_list
    
    
    def print_trees_leaf_low_majority(self,tree_list):
        """
        Analyzes and prints labels for the leaves of the trees based on majority rule.

        Args:
            tree_list (list): A list of trees whose leaves will be analyzed.

        Returns:
            numpy.ndarray: An array of labels indicating the majority label among the leaves.
        """
        list_alex = []
        # check if tree_list is a list
        if not isinstance(tree_list, list):
            tree_list = [tree_list]
        for tree in tree_list:
            local_list = []
            for node_up in tree.get_leaf_nodes():
                if not isinstance(node_up.name, str):
                    local_list.append(self.tmp[node_up.name])
                    continue
                for node_down in node_up.name.split("+"):
                    local_list.append(self.tmp[int(node_down)])
            sum_values = [sum(item) for item in zip(*local_list)]
            list_alex.append(sum_values)
            
        upgraded_list = []
        for i in range(len(list_alex[0])):
            majority = {}
            for j in range(len(list_alex)):
                if list_alex[j][i] != 0:
                    majority[j] = majority.get(j, 0) + list_alex[j][i]
            if not majority:
                upgraded_list.append(-1)
            else:
                upgraded_list.append(max(majority, key=majority.get))
        upgraded_list = np.array(upgraded_list, dtype=float)   
        return upgraded_list
    
    
    def print_trees_tree_wise_or(self,tree_list):
        """
        Processes and prints labels for entire trees using logical OR.

        Args:
            tree_list (list): A list of trees to be analyzed.

        Returns:
            numpy.ndarray: An array of labels processed using logical OR across each tree.
        """
        list_alex = []
        if not isinstance(tree_list, list):
            tree_list = [tree_list]
        for tree in tree_list:
            local_list = []
            for node in tree.root.get_all_parts_with_children_recurive():
                local_list.append(self.tmp[int(node)])
            # merge all labels together
            merged_list = [any(item) for item in zip(*local_list)]
            list_alex.append(merged_list)
        upgraded_list = []
        for i in range(len(list_alex[0])):
            for j in range(len(list_alex)):
                if list_alex[j][i] == 1:
                    upgraded_list.append(j)
                    break
            else:
                upgraded_list.append(-1)

        upgraded_list = np.array(upgraded_list, dtype=float)
        return upgraded_list
    
    def print_trees_tree_wise_majority(self,tree_list):
        """
        Processes and prints labels for entire trees based on majority rule.

        Args:
            tree_list (list): A list of trees to be analyzed.

        Returns:
            numpy.ndarray: An array indicating the majority label across each tree.
        """
        list_alex = []
        if not isinstance(tree_list, list):
            tree_list = [tree_list]
        for tree in tree_list:
            local_list = []
            for node in tree.root.get_all_parts_with_children_recurive():
                local_list.append(self.tmp[int(node)])
            # count the number of True values in each column and sum them up 
            sum_values = [sum(item) for item in zip(*local_list)]
            list_alex.append(sum_values)
        upgraded_list = []
        for i in range(len(list_alex[0])):
            majority = {}
            for j in range(len(list_alex)):
                if list_alex[j][i] != 0:
                    majority[j] = majority.get(j, 0) + list_alex[j][i]
            if not majority:
                upgraded_list.append(-1)
            else:
                upgraded_list.append(max(majority, key=majority.get))


        upgraded_list = np.array(upgraded_list, dtype=float)
        return upgraded_list
    
    def plot_tree(self, tree_list):
        """
        Initiates plotting for a list of trees, beginning from the root.

        Args:
            tree_list (list): A list of trees to be plotted.
        """
        """ Plot all trees in the list. """
        for tree in tree_list:
            if tree.root.name == "root":
                for child_node in tree.root.children:
                    self.plot_tree_recursive(child_node,True)
                    
            else:
                self.plot_tree_recursive(tree.root,True)
                

            
            
            

    def plot_tree_recursive(self, node,top = False):
        """
        Recursively plots a tree starting from the given node.

        Args:
            node (Custom_tree_node): The starting node for the plot.
            top (bool): Indicates if the node is the top node of its subtree.
        """
        """ Recursive function to plot a tree starting from a node. """
        
        current_node = node
        next_nodes = []
        parent_labels = []

        if isinstance(node.name, np.int64):
            parent_labels.append(node.name)
        else:
            for part in node.name.split("+"):
                parent_labels.append(int(part))
        if top and (not current_node.children):
            print("Complete solution for the current node: ", node.name)
            labels = self.labels_single_node(node)
            name = "Complete solution for the current node: " + str(node.name)
            self.plot_labels_3D(labels,name)
            self.plot_labels_2D(labels,name)
        for child in current_node.children:
            child_labels = []

            if isinstance(child.name, np.int64):
                child_labels.append(child.name)
            else:
                for subpart in child.name.split("+"):
                    child_labels.append(int(subpart))
            print("We are current comparing the following nodes: ", node.name, child.name, "hint the cluster number 2 are the combined stars of the two clusters")
            labels = self.compare_two_clusters_marke_shared(parent_labels, child_labels)
            name = "We are current comparing the following nodes: " + str(node.name) + str(child.name) + "hint the cluster number 2 are the combined stars of the two clusters"
            self.plot_labels_3D(labels,name)
            self.plot_labels_2D(labels,name)
            next_nodes.append(child)

        for next_node in next_nodes:
            self.plot_tree_recursive(next_node)

                
    
    def plot_labels_3D(self,labels, title = ""):
        """
        Plots the labels in a 3D space. HDR and Velocity

        Args:
            labels (numpy.ndarray): Labels to plot.
            title (str): Title for the plot.
        """
        # Plot the clusters
        fig = plot3D(labels, self.df_region, title, self.output_path, hrd= True, return_fig=True)
        fig.show()
    def plot_labels_2D(self,labels, title = ""):
        """
        Plots the labels in a 2D space.

        Args:
            labels (numpy.ndarray): Labels to plot.
            title (str): Title for the plot.
        """
        # Plot the clusters
        fig = plot2D(labels, self.df_region, title, self.output_path, hrd= True, return_fig=True)
        fig.show()