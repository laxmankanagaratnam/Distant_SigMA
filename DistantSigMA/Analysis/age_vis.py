import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import colors

from IsochroneArchive.myTools import my_utility

def create_cluster_id(label):
    return f'Orion_{label}'


# Paths
# ---------------------------------------------------------
# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
script_name = my_utility.get_calling_script_name(__file__)
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/', script_name=script_name)

run = "Orion_50_samples"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)


# df
Orion_df = pd.read_csv("/Users/alena/PycharmProjects/SigMA_Orion/Data/Sigma_200_clean_CMDs.csv")
Orion_df = Orion_df[Orion_df["cluster_label"] != -1]
cluster_names = [f'Orion_{num}' for num in np.unique(Orion_df["cluster_label"])]
# Apply the function to create the Cluster_id column
Orion_df.loc[:, 'Cluster_id'] = Orion_df.loc[:, 'cluster_label'].apply(lambda x: create_cluster_id(x))

ref_ages = "/Users/alena/PycharmProjects/SigMA_Orion/Data/Reference_ages_Orion_46.csv"
reference_ages = pd.read_csv(ref_ages)
# Rename the column
reference_ages.rename(columns={'Bona_fide_GRP': 'ref_age'}, inplace=True)

df_merged = pd.merge(Orion_df, reference_ages, on='Cluster_id')

# Drop rows with NaN values in the new column
df_merged.dropna(subset=['ref_age'], inplace=True)


plt.figure(figsize=(10, 6))
plt.scatter(df_merged['ra'], df_merged['dec'], c=df_merged['ref_age'], cmap='viridis')
plt.colorbar(label='Reference Age')
plt.xlabel('RA')
plt.ylabel('DEC')
plt.title('RA-DEC Plot with Color-Coded Reference Age')
plt.grid(True)
plt.show()

sorted_df = df_merged.sort_values(by=['cluster_label', 'ref_age'])
labels_orion = sorted_df["cluster_label"].to_numpy()
labels = labels_orion.reshape(len(labels_orion, ))

print(labels, "\n", len(np.unique(labels)))
icrs = True

if icrs:
    vel1 = "pmra"
    vel2 = "pmdec"
else:
    vel1 = "v_a_lsr"
    vel2 = "v_d_lsr"

cs = labels
df_plot = sorted_df.loc[cs != -1].reset_index(drop=True)
clustering_solution = cs.astype(int)
clustering_solution = clustering_solution[clustering_solution != -1]
cut_us = np.random.uniform(0, 1, size=clustering_solution.shape[0]) < 0.1

bg_opacity = 0.1
bg_color = '#383838'
#plt_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF',
#              '#FECB52', '#B82E2E', '#316395']


a = plt.get_cmap("RdBu_r")
norm = plt.Normalize(sorted_df.ref_age.min(), sorted_df.ref_age.max())
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])
age_range = sorted_df.drop_duplicates(subset="Cluster_id")
cm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm).to_rgba(age_range["ref_age"], alpha=None, bytes=False,
                                                               norm=True)
plt_colors = [colors.rgb2hex(c) for c in cm]



# Create figure
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
    column_widths=[0.7, 0.3],
    subplot_titles=['position', 'velocity'], )

# --------------- 3D scatter plot -------------------
trace_3d_bg = go.Scatter3d(
    x=df_plot.loc[cut_us, 'X'], y=df_plot.loc[cut_us, 'Y'], z=df_plot.loc[cut_us, 'Z'],
    mode='markers', marker=dict(size=1, color=bg_color, opacity=bg_opacity), hoverinfo='none', showlegend=False, )
fig.add_trace(trace_3d_bg, row=1, col=1)

trace_sun = go.Scatter3d(
    x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
    mode='markers', marker=dict(size=5, color='red', symbol='x'), hoverinfo='none', showlegend=True, name='Sun')
fig.add_trace(trace_sun, row=1, col=1)

# --------------- 3D cluster plot -------------------
for j, uid in enumerate(np.unique(clustering_solution)):
    if uid != -1:
        plot_points = (clustering_solution == uid)
        trace_3d = go.Scatter3d(
            x=df_plot.loc[plot_points, 'X'], y=df_plot.loc[plot_points, 'Y'], z=df_plot.loc[plot_points, 'Z'],
            mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)]), hoverinfo='none',
            showlegend=True, name=f'Cluster {np.unique(df_plot.loc[plot_points, "cluster_label"])[0]} / {np.unique(df_plot.loc[plot_points, "Chen"])[0]} / {np.unique(df_plot.loc[plot_points, "physical association"])[0]} ({np.sum(plot_points)} stars)', legendgroup=f'group-{uid}', )
        fig.add_trace(trace_3d, row=1, col=1)

# --------------- 2D vel plot -------------------
trace_vel_bg = go.Scatter(
    x=df_plot.loc[cut_us, vel1], y=df_plot.loc[cut_us, vel2],
    mode='markers', marker=dict(size=3, color=bg_color, opacity=bg_opacity), hoverinfo='none', showlegend=False)
fig.add_trace(trace_vel_bg, row=1, col=2)

for j, uid in enumerate(np.unique(clustering_solution)):
    if uid != -1:
        plot_points = (clustering_solution == uid)  # & cut_us
        trace_vel = go.Scatter(x=df_plot.loc[plot_points, vel1], y=df_plot.loc[plot_points, vel2],
                               mode='markers', marker=dict(size=3, color=plt_colors[j % len(plt_colors)], ),
                               hoverinfo='none', legendgroup=f'Cluster {np.unique(df_plot.loc[plot_points, "cluster_label"])}{np.unique(df_plot.loc[plot_points, "Chen"])[0]} /{np.unique(df_plot.loc[plot_points, "physical association"])[0]} ({np.sum(plot_points)} stars)', showlegend=False)
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
    legend=dict(itemsizing='constant',font=dict(
            color='white'
        )),
    # 3D plot
    scene=dict(
        xaxis=dict(xaxis),
        yaxis=dict(yaxis),
        zaxis=dict(zaxis)
    )
)

 # This will rearrange legend items to Trace A, Trace B, and Trace C

fig.write_html(output_path + f"test-o_age-2.html")

