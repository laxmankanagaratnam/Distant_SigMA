import os
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import DistantSigMA.misc.utilities as ut
import numpy as np
from scipy.stats import gaussian_kde, skew

matplotlib.use('TkAgg')

# ---------------------------------------------------------
# 1. Set output path + load data
# ---------------------------------------------------------

output_path = ut.set_output_path(script_name="DeNoise")

run = "Task_2_preRemoval_diagnostics"
output_path = output_path + f"{run}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)


sns.set(style="whitegrid")


df = pd.read_csv(output_path+"SigMA_clustering_aligned.csv")

group_column = "labels_SigMA_aligned"
value_column = "density"

# Store results
plot_data = []
skewness_df = pd.DataFrame(columns=["labels_SigMA_aligned", "skewness", "stars_over_bg"])

i = 0
for label, group in df.groupby(group_column):
    values = group[value_column].dropna().values

    kde = gaussian_kde(values)
    x_vals = np.linspace(min(values), max(values), 200)
    y_vals = kde(x_vals)

    y_vals_norm = y_vals / y_vals.max()  # Normalize density to 1
    group_skew = skew(values)

    for x, y in zip(x_vals, y_vals_norm):
        plot_data.append({
            "x": x,
            "density": y,
            "label": label,
            "skewness": group_skew
        })

    if label in group["reference"].values:
        star_over_bg = (group["reference"] == label).sum() / (group["reference"] != label).sum()
    else:
        star_over_bg = (group["reference"] != -1).sum() / (group["reference"] == -1).sum()

    skewness_df.loc[i, "labels_SigMA_aligned"] = label
    skewness_df.loc[i, "skewness"] = group_skew
    skewness_df.loc[i, "stars_over_bg"] = star_over_bg
    i+=1

# --- Step 2: Create a DataFrame from precomputed results ---
kde_df = pd.DataFrame(plot_data)

# Sort label order by skewness
label_order = (
    kde_df.groupby("label")["skewness"]
    .mean()
    .sort_values()
    .index.tolist()
)

# --- Step 3: Plot using FacetGrid ---
g = sns.FacetGrid(
    kde_df,
    col="label",
    col_wrap=5,
    col_order=label_order,
    sharex=True,
    sharey=True,
    height=2.5,
    aspect=1.2
)


def facet_kde(data, color, **kwargs):
    ax = plt.gca()
    ax.plot(data["x"], data["density"], color=color)
    skew_val = data["skewness"].iloc[0]
    label = data["label"].iloc[0]
    ax.set_title(f"{label}\nskew={skew_val:.2f}", fontsize=9)


g.map_dataframe(facet_kde)
g.set_axis_labels(value_column, "Normalized density")
plt.tight_layout()
plt.close()


print(skewness_df)

f, ax = plt.subplots(1,1)
ax.scatter(skewness_df.stars_over_bg, skewness_df.skewness)
ax.set_aspect("equal")
plt.show()