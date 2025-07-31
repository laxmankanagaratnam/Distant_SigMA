import pandas as pd
import numpy as np
from functools import reduce
import operator
import os

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import DistantSigMA.misc.utilities as ut


def density_slider_plot(cluster_id, df, label_list, observable: str, observable_df, pos_params=None):
    plt_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#FF6692', '#B6E880', '#FF97FF',
                  '#FECB52', '#B82E2E', '#316395']

    if pos_params is None:
        pos_params = ["X", "Y", "Z"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scatter3d"}]],
        column_widths=[0.35, 0.65],
        subplot_titles=['Knee-diagram', '3D Position']
    )

    # Define and sort data
    found_cluster = df[df[label_list[0]] == cluster_id]
    found_cluster = found_cluster.sort_values(by="density")

    # Set density thresholds for slider steps
    density_thresholds =observable_df.loc[:,f"{observable}_x"].to_numpy()

    # 2D line plot: cumulative count vs. threshold
    line2d = go.Scatter(
        x=observable_df.loc[:,f"{observable}_x"],
        y=observable_df.loc[:,f"{observable}_y"],
        mode='lines',
        name=f'{observable}',
        line=dict(color=plt_colors[cluster_id % len(plt_colors)]),
        showlegend=True
    )
    fig.add_trace(line2d, row=1, col=1)

    # 3D background scatter plot (all stars in cluster)
    bg_3d = go.Scatter3d(
        x=found_cluster[pos_params[0]],
        y=found_cluster[pos_params[1]],
        z=found_cluster[pos_params[2]],
        mode='markers',
        name="Noise",
        legendgroup="background",
        marker=dict(size=2, color="darkgray"),
        hoverinfo='text',
        showlegend=True
    )
    fig.add_trace(bg_3d, row=1, col=2)

    # Static legend entry for the cluster (shown only once)
    legend_trace = go.Scatter3d(
        x=[None], y=[None], z=[None],  # dummy point
        mode='markers',
        marker=dict(size=4, color=plt_colors[cluster_id % len(plt_colors)]),
        name=f"Cluster {cluster_id}",
        showlegend=True,
        legendgroup="cluster"
    )
    fig.add_trace(legend_trace, row=1, col=2)


    # Save index of static traces
    initial_traces = len(fig.data)

    # Slider steps
    steps = []

    for step_idx, rho in enumerate(density_thresholds):
        # Update labels based on current threshold
        df_tmp = df.copy()
        mask = (df_tmp[label_list[0]] == cluster_id) & (df_tmp.density < rho)
        df_tmp.loc[mask, label_list[0]] = -1

        cs = df_tmp[label_list[0]]
        df_plot = df_tmp.loc[cs != -1]
        clustering_solution = cs.astype(int)
        plot_points = (clustering_solution == cluster_id)

        # 3D scatter for current step
        trace_3d = go.Scatter3d(
            x=df_plot.loc[plot_points, 'X'],
            y=df_plot.loc[plot_points, 'Y'],
            z=df_plot.loc[plot_points, 'Z'],
            mode='markers',
            marker=dict(size=3, color=plt_colors[cluster_id % len(plt_colors)]),
            hoverinfo='none',
            name=f"Cluster {cluster_id}",
            showlegend=False
        )
        fig.add_trace(trace_3d, row=1, col=2)

        # 2D vertical line marker for current density threshold
        trace_vline = go.Scatter(
            x=[rho, rho],
            y=[0, observable_df.loc[:,f"{observable}_y"].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Threshold',
            showlegend=False
        )
        fig.add_trace(trace_vline, row=1, col=1)

        # Build visibility map
        n_dynamic = len(density_thresholds) * 2
        visibility = [True] * initial_traces + [False] * n_dynamic
        visibility[initial_traces + step_idx * 2] = True       # 3D cluster points
        visibility[initial_traces + step_idx * 2 + 1] = True   # 2D vertical line

        steps.append(dict(
            method='update',
            args=[{'visible': visibility},
                  {'title': f'Density Threshold: {rho:.2f}'}],
            label=f'{rho:.2f}'
        ))

    # Set initial trace visibility: only static traces and first step's dynamic traces
    for i, trace in enumerate(fig.data):
        if i < initial_traces:
            trace.visible = True
        else:
            step_idx = (i - initial_traces) // 2
            trace.visible = (step_idx == 0)

    # Add slider to layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Density threshold: "},
        pad={"t": 50},
        steps=steps
    )]

    # Axis styling
    fig.update_yaxes(title_text=f"{observable}", showgrid=True, row=1, col=1)
    fig.update_xaxes(title_text="Density", showgrid=True, row=1, col=1)

    # Finalize layout
    fig.update_layout(
        sliders=sliders,
        title="",
        width=1200,
        height=700,
        showlegend=True,
        legend=dict(itemsizing='constant')
    )

    return fig

def analyze_cluster(cluster_id, df, label_list, cmd_params, pos_params=None, vel_params=None):
    plt_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#FF6692', '#B6E880', '#FF97FF',
                  '#FECB52', '#B82E2E', '#316395']

    if vel_params is None:
        vel1 = "v_a_lsr"
        vel2 = "v_d_lsr"
    else:
        vel1, vel2 = vel_params

    if pos_params is None:
        pos_params = ["X", "Y", "Z"]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.4, 0.3, 0.3],
        subplot_titles=['position', 'velocity', 'CMD'], )

    condition = reduce(operator.or_, (df[label] == cluster_id for label in label_list))

    # Apply condition
    union = df[condition]

    # Bg in plots
    # 3D
    bg_3d = go.Scatter3d(
        x=union.loc[:, pos_params[0]],
        y=union.loc[:, pos_params[1]],
        z=union.loc[:, pos_params[2]],
        mode='markers',
        name="background",
        legendgroup="background",
        marker=dict(size=2, color="darkgray"),
        hoverinfo='text',
        showlegend=True
    )
    fig.add_trace(bg_3d, row=1, col=1)

    # 2D vel
    bg_vel = go.Scatter(
        x=union.loc[:, vel1],
        y=union.loc[:, vel2],
        mode='markers',
        marker=dict(size=2, color="darkgray"),
        hoverinfo='none',
        name="background",
        legendgroup="background",
        showlegend=False
    )
    fig.add_trace(bg_vel, row=1, col=2)

    # CMD
    bg_cmd = go.Scatter(
        x=union.loc[:, cmd_params[0]],
        y=union.loc[:, cmd_params[1]],
        mode='markers',
        marker=dict(size=2, color="darkgray"),
        hoverinfo='none',
        name="background",
        legendgroup="background",
        showlegend=False
    )
    fig.add_trace(bg_cmd, row=1, col=3)

    # Variable plots
    true_mask = df[label_list[0]] == cluster_id

    for i in range(len(label_list) + 1):

        if i < len(label_list):
            label = label_list[i]
            pred_mask = df[label_list[i]] == cluster_id
        else:
            label = "intersection"
            pred_mask = (df[label_list[1:]] == cluster_id).all(axis=1)

        TP = (true_mask & pred_mask).sum()
        FP = (~true_mask & pred_mask).sum()
        FN = (true_mask & ~pred_mask).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        contamination = 1 - precision

        # Subset data for this cluster in this strategy
        cluster_df = df[pred_mask]

        # Build hover text per point
        hover_text = [
                         f"Precision: {precision:.2f}<br>"
                         f"Recall: {recall:.2f}<br>"
                         f"Contamination: {contamination:.2f}"
                     ] * len(cluster_df)

        # 3D SCATTER
        # ------------
        cluster_pos = go.Scatter3d(
            x=cluster_df.loc[:, pos_params[0]],
            y=cluster_df.loc[:, pos_params[1]],
            z=cluster_df.loc[:, pos_params[2]],
            mode='markers',
            marker=dict(size=4, color=plt_colors[i]),
            name=f"{label}",
            legendgroup=f"{label}",
            text=hover_text,
            hoverinfo='text',
            showlegend=True
        )
        fig.add_trace(cluster_pos, row=1, col=1)

        # 2D scatter
        # -----------
        cluster_vel = go.Scatter(
            x=cluster_df.loc[:, vel1],
            y=cluster_df.loc[:, vel2],
            mode='markers',
            marker=dict(size=4, color=plt_colors[i]),
            text=hover_text,
            hoverinfo='text',
            name=f"{label}",
            legendgroup=f"{label}",
            showlegend=False
        )
        fig.add_trace(cluster_vel, row=1, col=2)

        # CMD
        # -----------
        cluster_cmd = go.Scatter(
            x=cluster_df.loc[:, cmd_params[0]],
            y=cluster_df.loc[:, cmd_params[1]],
            mode='markers',
            marker=dict(size=4, color=plt_colors[i]),
            text=hover_text,
            hoverinfo='text',
            name=f"{label}",
            legendgroup=f"{label}",
            showlegend=False,
        )
        fig.add_trace(cluster_cmd, row=1, col=3)

    # ------------ Update axis information ---------------
    # 3d position
    plt_kwargs = dict(showbackground=False, showline=False, zeroline=True, zerolinecolor='white', zerolinewidth=2,
                      showgrid=True, showticklabels=True, color="white",
                      linecolor='white', linewidth=1, gridcolor='white')

    # pos
    fig.update_yaxes(title_text="X", showgrid=True, row=1, col=1, )
    fig.update_yaxes(title_text="Y", showgrid=True, row=1, col=1, )
    fig.update_yaxes(title_text="Z", showgrid=True, row=1, col=1, )

    # vel
    fig.update_yaxes(title_text=vel1, showgrid=True, row=1, col=2, )
    fig.update_yaxes(title_text=vel2, showgrid=True, row=1, col=2, )
    # CMD
    fig.update_yaxes(title_text="color index", showgrid=True, row=1, col=3, )
    fig.update_yaxes(title_text="abs mag", range=[df[cmd_params[1]].max(), df[cmd_params[1]].min()], showgrid=True,
                     row=1, col=3, )

    # Finalize layout
    fig.update_layout(
        title="",
        # width=800,
        # height=800,
        showlegend=True,
        # paper_bgcolor='#383838',
        # plot_bgcolor='#383838',
        legend=dict(itemsizing='constant'),
    )

    return fig


if __name__ == "__main__":

    # Paths
    output_path = ut.set_output_path(script_name="noise_removal_visualization")

    run = "Density_slider"
    output_path = output_path + f"{run}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_csv("datafiles/SigMA_pruned.csv")
    df.loc[:, "scaled_parallax"] = df["parallax"] * 0.2

    # # label_list = ["real_ref", "labels_SigMA", "labels_SigMA_noisy", "nstar_labels", "v_disp_labels", "MCD_labels"]
    # label_list = ["real_ref", "SigMA_noisy", "MCD_labels", "Nstar_labels", "velocity_labels"]
    #
    # df["abs_G"] = df["phot_g_mean_mag"] + 5 * np.log10(df["parallax"]) - 10
    # df["bprp"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    #
    # f = analyze_cluster(cluster_id=1,
    #                     df=df,
    #                     label_list=label_list,
    #                     cmd_params=["bprp", "abs_G"], pos_params=["ra", "dec", "scaled_parallax"],
    #                     vel_params=["pmra", "pmdec"])
    #
    # f.write_html(output_path + "DeNoise_equal_icrs.html")

    obs_df = pd.read_csv("datafiles/observables_cluster_1.csv")

    f = density_slider_plot(1, df=df, label_list=["SigMA_noisy_pruned"], observable="MCD", observable_df=obs_df)


    f.write_html(output_path + "density_slider_MCD.html")
