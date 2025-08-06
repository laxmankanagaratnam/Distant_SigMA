from sklearn.covariance import MinCovDet
from sklearn.metrics import f1_score


def deNoise_MCD(df, label_col,
                reference_col=None,
                parameters=None, support_frac=0.55):

    if parameters is None:
        parameters = ["ra", "dec", "scaled_parallax", "pmra", "pmdec", "density"]

    # Initialize the MCD column if it doesn't exist
    df['MCD'] = df[label_col]

    f1_scores = {}

    for label, group in df.groupby(label_col):
        if len(group) < len(parameters) + 1:
            # Not enough data points to fit MCD
            f1_scores[label] = float('nan')
            continue

        X = group[parameters].to_numpy()

        try:
            mcd = MinCovDet(support_fraction=support_frac).fit(X)
        except Exception as e:
            f1_scores[label] = float('nan')
            continue

        mcd_mask = mcd.support_

        # Index of group in original df
        group_indices = group.index

        # Default all points in group to the label
        df.loc[group_indices, 'MCD'] = label

        # Set outliers to -1
        outlier_indices = group_indices[~mcd_mask]
        df.loc[outlier_indices, 'MCD'] = -1

        # Compute F1 score
        if reference_col:
            y_true_bin = (df.loc[group_indices, reference_col] == label).astype(int)
            y_pred_bin = (df.loc[group_indices, 'MCD'] == label).astype(int)

            score = f1_score(y_true_bin, y_pred_bin, average='binary')
            f1_scores[label] = score

    return f1_scores


if __name__ == "__main__":

    import os
    import pandas as pd
    from sklearn.covariance import MinCovDet
    import matplotlib
    from scipy.sparse import load_npz
    from sklearn.metrics import f1_score
    import DistantSigMA.misc.utilities as ut

    matplotlib.use('TkAgg')

    # ---------------------------------------------------------
    # 1. Set output path + load data
    # ---------------------------------------------------------

    output_path = ut.set_output_path(script_name="DeNoise")

    run = "Task_4_MCD"
    output_path = output_path + f"{run}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load data
    data = pd.read_csv(output_path + "SigMA_clustering_aligned.csv")

    fscores = deNoise_MCD(
        df=data,
        label_col="labels_SigMA_aligned",
        reference_col="reference"

    )

    print(fscores)