import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, normalized_mutual_info_score as nmi
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


def encode_labels_by_frequency(series):
    # Get label counts, sorted descending
    label_counts = series.value_counts()

    # Create mapping
    label_map = {label: i - 1 for i, label in enumerate(label_counts.index)}

    # Apply the mapping
    return series.map(label_map)  # , label_map


def OLD_map_ref_to_sigma(input_data, ref_label, label2match):
    """
    HUGE CAVEAT: This mapping may collapse SigMA clusters if both share the same reference label as max label frequency
    """
    data = input_data.copy()

    # Create a mapping from each real_SigMA label to the most frequent real_ref label in that group
    label_mapping = (
        data.groupby(label2match)[ref_label]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    # Apply the mapping to update the real_SigMA column
    data[label2match] = data[label2match].map(label_mapping)


def align_ref_to_sigma(input_data, ref_label, label2match, new_col_name):
    """
    NEW version of "map_ref_to_sigma" that should fix the previous caveats
    """
    data = input_data.copy()

    # Ensure labels are integers
    data[ref_label] = data[ref_label].astype(int)
    data[label2match] = data[label2match].astype(int)

    # Get the unique labels from both
    ref_labels = np.sort(data[ref_label].unique())
    pred_labels = np.sort(data[label2match].unique())

    # Use union of both label sets to avoid shape mismatch
    all_labels = np.unique(np.concatenate([ref_labels, pred_labels]))

    # Step 1: Compute confusion matrix using full label set
    cm = confusion_matrix(data[ref_label], data[label2match], labels=all_labels)

    # Step 2: Apply Hungarian Algorithm to maximize agreement
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Step 3: Create mapping from predicted label → reference label
    mapping = {all_labels[col]: all_labels[row] for row, col in zip(row_ind, col_ind)}

    # Step 4: Apply mapping to label2match column
    data[new_col_name] = data[label2match].map(mapping)

    return data


def analyze_solution(data, reference_labels, labels2compare, title: str = "Confusion Matrix"):
    report = classification_report(data[reference_labels], data[labels2compare], digits=3, output_dict=True)

    df_report = pd.DataFrame(report).transpose()

    # Specify labels (as we start with -1 for noise)
    all_labels = np.union1d(data[reference_labels].unique(), data[labels2compare].unique())
    all_labels_sorted = sorted(all_labels)

    # Compute confusion matrix with fixed label order
    conf_mat = confusion_matrix(
        data[reference_labels],
        data[labels2compare],
        labels=all_labels_sorted
    )

    # Normalize by row to get percentages
    conf_mat_percent = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True) * 100

    # Plot with log-normalized color scale
    f, ax = plt.subplots(1, 1, figsize=(12, 10))
    sns.heatmap(
        conf_mat_percent,
        xticklabels=all_labels_sorted,
        yticklabels=all_labels_sorted,
        cmap="Blues",
        norm=LogNorm(vmin=1, vmax=conf_mat_percent.max()),  # Avoid log(0) with vmin=1
        cbar_kws={'label': 'log(count)'}
    )
    for y in range(conf_mat_percent.shape[0]):
        for x in range(conf_mat_percent.shape[1]):
            value = conf_mat_percent[y, x]  # <-- fixed
            if value > 0:
                ax.text(x + 0.5, y + 0.5, f"{value:.1f}",
                        ha='center', va='center', color='black', fontsize=9)

    plt.xlabel("Predicted (SigMA)")
    plt.ylabel("True (Reference)")
    plt.title(title)

    return df_report, f


def noisy_solution(p, input_data, reference_labels, clustered_labels, new_label_col: str = "labels_noisy"):
    data = input_data.copy()

    # Filter the rows with -1 in both label columns
    mask = (data[reference_labels] == -1) & (data[clustered_labels] == -1)
    noise_df = data[mask]

    # Determine how many rows to change and randomly sample them from noise_df
    n_to_change = int(len(noise_df) * p)
    noise_to_cluster = noise_df.sample(n=n_to_change, random_state=42)

    # Get list of possible labels from SigMA solution (excluding -1)
    possible_labels = data[clustered_labels].unique()
    possible_labels = [lbl for lbl in possible_labels if lbl != -1]

    # Assign a random label from possible_labels to 'labels_SigMA' in those rows
    data.loc[:, new_label_col] = data[clustered_labels]
    data.loc[noise_to_cluster.index, new_label_col] = np.random.choice(possible_labels, size=n_to_change)

    return data


if __name__ == "__main__":
    input_data = pd.read_csv(
        "/DistantSigMA/misc/noise_removal/datafiles/noise_removal_input.csv",
        #   usecols=["labels_SigMA", "reference", "density", "v_a_lsr", "v_d_lsr"]
    )
    input_data["real_ref"] = encode_labels_by_frequency(input_data.reference)

    input_data.sort_values(by="reference", inplace=True)
    print(np.unique(input_data.reference, return_counts=True))

    input_data.sort_values(by="labels_SigMA", inplace=True)

    print(np.unique(input_data.labels_SigMA, return_counts=True))

    data = align_ref_to_sigma(input_data=input_data, ref_label="real_ref", label2match="labels_SigMA",
                              new_col_name="labels_SigMA_aligned")

    data.sort_values(by="real_ref", inplace=True)

    print("NMI: ", nmi(data.real_ref, data.labels_SigMA_aligned))
    print(np.unique(data.real_ref, return_counts=True), "\n \n",
          np.unique(data.labels_SigMA_aligned, return_counts=True))

    # ---------------------------------------------------------
    # 2. Artificially add noise to the labels
    #
    #    by assigning stars that were classified as -1 in both
    #    the reference and SigMA labelling randomly to a given
    #    cluster in the SigMA labels
    # ---------------------------------------------------------

    # data = noisy_solution(p=0.5,
    #                       input_data=input_data,
    #                       reference_labels="reference",
    #                       clustered_labels="labels_SigMA_aligned",
    #                       new_label_col="SigMA_noisy")
    #
    # data.sort_values(by="SigMA_noisy", inplace=True)
