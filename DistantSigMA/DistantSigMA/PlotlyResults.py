import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.interpolate import griddata


def create_grid_and_interpolate(x, y, z):
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    return grid_x, grid_y, grid_z


def plot(labels: np.array, df: pd.DataFrame, filename: str, output_pathname: str = None, hrd: bool = False,
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


def plot_darkmode(labels: np.array, df: pd.DataFrame, filename: str, output_pathname: str = None, hrd: bool = False,
                  icrs: bool = False, return_fig: bool = False):
    """ Simple function for creating a result plot of all the final clusters in Dark mode."""

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
    plt_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#FF6692', '#B6E880', '#FF97FF',
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
    plt_kwargs = dict(showbackground=False, showline=False, zeroline=True, zerolinecolor='white', zerolinewidth=2,
                      showgrid=True, showticklabels=True, color="white",
                      linecolor='white', linewidth=1, gridcolor='white')

    xaxis = dict(**plt_kwargs, title='X [pc]')  # , tickmode = 'linear', dtick = 50, range=[-50,200])
    yaxis = dict(**plt_kwargs, title='Y [pc]')  # , tickmode = 'linear', dtick = 50, range=[-200, 50])
    zaxis = dict(**plt_kwargs, title='Z [pc]')  # , tickmode = 'linear', dtick = 50, range=[-100, 150])

    # tangential vel
    if not icrs:
        fig.update_xaxes(title_text="v_alpha", showgrid=False, row=1, col=2, color="white")
        fig.update_yaxes(title_text="v_delta", showgrid=False, row=1, col=2, color="white")
    else:
        fig.update_xaxes(title_text="pmra", showgrid=False, row=1, col=2, color="white")
        fig.update_yaxes(title_text="pmdec", showgrid=False, row=1, col=2, color="white")
    # tangential vel

    # Finalize layout
    fig.update_layout(
        title="",
        # width=800,
        # height=800,
        showlegend=True,
        paper_bgcolor='#383838',
        plot_bgcolor='#383838',
        legend=dict(itemsizing='constant'),
        # 3D plot
        scene=dict(
            xaxis=dict(xaxis),
            yaxis=dict(yaxis),
            zaxis=dict(zaxis)
        )
    )

    if output_pathname:
        fig.write_html(output_pathname + f"{filename}.html")

    if return_fig:
        return fig


def plot_surface(labels: np.array, df: pd.DataFrame, filename: str, output_pathname: str = None,
                 icrs: bool = False, return_fig: bool = False):
    """ Simple function for creating a result plot of all the final clusters in Dark mode."""

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
    plt_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#FF6692', '#B6E880', '#FF97FF',
                  '#FECB52', '#B82E2E', '#316395']

    #  ---------------  Create figure  ---------------
    #  --------------- ---------------  --------------
    fig_pos = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scatter3d"}]],
        subplot_titles=['position'], )

    fig_vel = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "xy"}]],
        subplot_titles=['velocity'], )

    fig_HRD = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "xy"}]],
        subplot_titles=['HRD'], )

    # --------------- 3D scatter plot -------------------

    # sun marker
    trace_sun = go.Scatter3d(
        x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
        mode='markers', marker=dict(size=5, color='red', symbol='x'), hoverinfo='none', showlegend=True, name='Sun')
    fig_pos.add_trace(trace_sun, row=1, col=1)

    # 3D clusters
    for j, uid in enumerate(np.unique(clustering_solution)):  # for each cluster label in the label array
        if uid != -1:
            plot_points = (clustering_solution == uid)  # grab the right locations
            trace_surface = go.Mesh3d(x=df_plot.loc[plot_points, 'X'], y=df_plot.loc[plot_points, 'Y'],
                                      z=df_plot.loc[plot_points, 'Z'], color=plt_colors[j % len(plt_colors)],
                                      opacity=0.5, alphahull=7,
                                      showlegend=True, name=f'Cluster {int(uid)} ({np.sum(plot_points)} stars)',
                                      legendgroup=f'group-{uid}', )
            fig_pos.add_trace(trace_surface, row=1, col=1)  # add cluster trace

    # --------------- 2D vel plot -------------------

    # cluster velocities (same as for 3D positions)
    for j, uid in enumerate(np.unique(clustering_solution)):
        if uid != -1:
            plot_points = (clustering_solution == uid)  # & cut_us
            trace_vel = go.Scatter(x=df_plot.loc[plot_points, vel1], y=df_plot.loc[plot_points, vel2],
                                   mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)], ),
                                   hoverinfo='none', legendgroup=f'group-{uid}',
                                   name=f'Cluster {uid} ({np.sum(plot_points)} stars)', showlegend=True)
            fig_vel.add_trace(trace_vel, row=1, col=1)

    # ------------ Update axis information ---------------
    # 3d position
    plt_kwargs = dict(showbackground=False, showline=False, zeroline=True, zerolinecolor='white', zerolinewidth=2,
                      showgrid=True, showticklabels=True, color="white",
                      linecolor='white', linewidth=1, gridcolor='white')

    xaxis = dict(**plt_kwargs, title='X [pc]')  # , tickmode = 'linear', dtick = 50, range=[-50,200])
    yaxis = dict(**plt_kwargs, title='Y [pc]')  # , tickmode = 'linear', dtick = 50, range=[-200, 50])
    zaxis = dict(**plt_kwargs, title='Z [pc]')  # , tickmode = 'linear', dtick = 50, range=[-100, 150])

    # Finalize layout
    fig_pos.update_layout(
        title="",
        # width=800,
        # height=800,
        title_font=dict(color="white"),
        font=dict(color="white"),
        showlegend=True,
        paper_bgcolor='#383838',
        plot_bgcolor='#383838',
        legend=dict(itemsizing='constant', font=dict(color="white")),
        # 3D plot
        scene=dict(
            xaxis=dict(xaxis),
            yaxis=dict(yaxis),
            zaxis=dict(zaxis)
        )
    )

    # tangential vel
    if not icrs:
        fig_vel.update_xaxes(title_text="v_alpha", showgrid=False, row=1, col=1, color="white")
        fig_vel.update_yaxes(title_text="v_delta", showgrid=False, row=1, col=1, color="white")
    else:
        fig_vel.update_xaxes(title_text="pmra", showgrid=False, row=1, col=1, color="white")
        fig_vel.update_yaxes(title_text="pmdec", showgrid=False, row=1, col=1, color="white")

    fig_vel.update_layout(
        title="",
        title_font=dict(color="white"),
        font=dict(color="white"),
        showlegend=True,
        paper_bgcolor='#383838',
        plot_bgcolor='#383838',
        legend=dict(itemsizing='constant', font=dict(color="white")),
        )

    # --------------- HRD plot -------------------

    # HRD of each cluster
    for j, kid in enumerate(np.unique(clustering_solution)):
        if kid != -1:
            plot_points = (clustering_solution == kid)  # & cut_us
            trace_hrd = go.Scatter(x=df_plot.loc[plot_points, 'g_rp'], y=df_plot.loc[plot_points, 'mag_abs_g'],
                                   mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)], ),
                                   hoverinfo='none', legendgroup=f'group-{kid}',
                                   name=f'Cluster {kid} ({np.sum(plot_points)} stars)', showlegend=True)
            fig_HRD.add_trace(trace_hrd, row=1, col=1)

    fig_HRD.update_xaxes(title_text="G-RP", showgrid=False, row=1, col=1, color = "white")
    fig_HRD.update_yaxes(title_text="Abs mag G", showgrid=False, autorange="reversed", row=1, col=1, color="white")

    # Finalize layout
    fig_HRD.update_layout(
            title="",
            title_font=dict(color="white"),
            font=dict(color="white"),
            showlegend=True,
            paper_bgcolor='#383838',
            plot_bgcolor='#383838',
            legend=dict(itemsizing='constant', font=dict(color="white")),
        )


    if output_pathname:
        fig_pos.write_html(output_pathname + f"{filename}_position.html")
        fig_vel.write_html(output_pathname + f"{filename}_velocity.html")
        fig_HRD.write_html(output_pathname + f"{filename}_HRD.html")

    if return_fig:
        return fig_pos, fig_vel, fig_HRD


