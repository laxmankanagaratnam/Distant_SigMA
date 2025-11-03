import pandas as pd
from astropy.io import fits
import numpy as np
from typing import Union
from scipy.stats import median_abs_deviation as MAD
from SigMA.bayesian_velocity_scaling import scale_factors as sf_function


def setup_Cartesian_ps(df_fit: pd.DataFrame, KNN_list: list, beta: float, knn_initcluster_graph: int,
                       info_path: str = None, cluster_features: list = None, nb_resampling: int = 10,
                       kd_tree_data=None):
    """
    Function that automatically sets up the Cartesian phase-space.

    :param df_fit: Input dataframe
    :param KNN_list: List of nearest neighbor values to loop through
    :param beta: beta value
    :param knn_initcluster_graph: knn_initcluster_value
    :return: dictionary of setup parameters
    """

    # set cluster parameters specific to the chosen coordinate system
    n_resampling = nb_resampling
    if cluster_features is None:
        cluster_features = ['X', 'Y', 'Z', 'v_a_lsr', 'v_d_lsr']
    # scaling relationship
    scale_factor_list, mean_sf, scale_factors = bayesian_scaling(df_fit=df_fit, info_path=info_path,
                                                                 cols=cluster_features[3:])
    # SigMA kwargs
    sigma_kwargs = dict(cluster_features=cluster_features, scale_factors=scale_factors, nb_resampling=n_resampling,
                        max_knn_density=max(KNN_list) + 1, beta=beta, knn_initcluster_graph=knn_initcluster_graph,
                        kd_tree_data=kd_tree_data)

    setup_dict = {"scale_factor_list": scale_factor_list, "mean_sf": mean_sf, "scale_factors": scale_factors,
                  "sigma_kwargs": sigma_kwargs}
    return setup_dict


def setup_ICRS_ps(df_fit: pd.DataFrame, sf_params: Union[str, list], sf_range: Union[list, np.linspace, range],
                  KNN_list: Union[list, np.linspace, range], beta: float, knn_initcluster_graph: int,
                  scaling: str = None, means=None, kd_tree_data=None):
    """
    Function that automatically sets up the ICRS phase-space.


    :param df_fit: Input dataframe
    :param KNN_list: List of nearest neighbor values to loop through
    :param beta: beta value
    :param knn_initcluster_graph: knn_initcluster_value
    :param sf_params: parameter name or list of parameter names
    :param sf_range: scaling factor range/linspace or list of scaling factor ranges/linspaces.
    :param scaling: str or none
    :return: dictionary of setup parameters and the scaled dataframe
    """

    # resampling is not implemented for ICRS yet
    n_resampling = 0

    # scale the ICRS parameters
    if type(scaling) == str:

        cols_to_scale = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
        scaled_cols = ['ra_scaled', 'dec_scaled', 'parallax_scaled', 'pmra_scaled', 'pmdec_scaled']

        df_scaled = df_fit.copy()
        scaled_data = [parameter_scaler(df_scaled[col], scaling) for col in cols_to_scale]
        for col_id, col in enumerate(scaled_cols):
            df_scaled[col] = scaled_data[col_id]
        df_fin = df_scaled

        if kd_tree_data is not None:
            df_kd_tree_data_scaled = kd_tree_data.copy()
            scaled_kd_tree_data = [parameter_scaler(df_kd_tree_data_scaled[col], scaling) for col in cols_to_scale]
            for col_id, col in enumerate(scaled_cols):
                df_kd_tree_data_scaled[col] = scaled_kd_tree_data[col_id]
        else:
            df_kd_tree_data_scaled = None


    else:
        scaled_cols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
        df_fin = df_fit
        # TODO: Implement scaling here for kdtree_data
        df_kd_tree_data_scaled = kd_tree_data

    if type(sf_params) == str:
        mean_sf, scale_factors = single_parameter_scaling(sf_params, sf_range)
    elif type(sf_params) == list:
        scale_factors = means
    else:
        raise ValueError("SF-Param input wrong")

    # SigMA kwargs
    sigma_kwargs = dict(cluster_features=scaled_cols, scale_factors=scale_factors, nb_resampling=n_resampling,
                        max_knn_density=max(KNN_list) + 1, beta=beta, knn_initcluster_graph=knn_initcluster_graph,
                        kd_tree_data=df_kd_tree_data_scaled)

    setup_dict = {"scale_factor_list": sf_range, "scale_factors": scale_factors, "sigma_kwargs": sigma_kwargs}

    return setup_dict, df_fin


def parameter_scaler(input_arr: np.array, scaling: str):
    """
    Function that scales an input variable.

    :param input_arr: unscaled parameter
    :param scaling:  uses the median and mad as scalers, the mean and std, or does not scale at all
    :return: scaled parameter
    """
    if scaling == "robust":
        return (input_arr - np.median(input_arr)) / MAD(input_arr)
    elif scaling == "normal":
        print("Scaling not robust.")
        return (input_arr - np.mean(input_arr)) / np.std(input_arr)
    else:
        return input_arr


def bayesian_scaling(df_fit: pd.DataFrame, info_path: str, cols=None):
    """
    Bayesian calculation of the scaling factors for the phase-space velocity sub-space. For now only in use for
    Cartesian phase-space.

    :param df_fit: Input data
    :param info_path: path to the .npz file
    :return: List of 10 scale factors (floats), the mean of this list and the dictionary to pass to the SigMA instance
    """
    if info_path is None:
        info_path = '../Data/bayesian_LR_data.npz'

    if cols is None:
        cols = ['v_a_lsr', 'v_d_lsr']
    vel_scaling_info = np.load(info_path)
    x_sf, post_pred_sf = vel_scaling_info['x'], vel_scaling_info['posterior_predictive']
    d = 1000 / df_fit.parallax
    scale_factor_list = sf_function(x_sf, post_pred_sf, x_range=(d.min(), d.max()))
    mean_sf = np.mean(scale_factor_list)
    scale_factors = {'vel': {'features': cols, 'factor': mean_sf}}
    return scale_factor_list, mean_sf, scale_factors


def single_parameter_scaling(param: str, scaling_range: Union[range, np.linspace]):
    """
    Define scale factors for a single parameter. Default is the mean of the range.
    :param param: Parameter name
    :param scaling_range: range or linspace of scaling factors
    :return: mean scaling factor, scale_factor dictionary
    """

    mean_sf = np.mean(scaling_range)
    scale_factors = {'pos': {'features': [param], 'factor': mean_sf}}

    return mean_sf, scale_factors


def multiple_parameter_scaling(param_list: list, scaler_list: list):
    """
    Define scale factors for multiple features.

    :param param_list: list of parameter names (= features)
    :param scaler_list: list of scaling values (N= N_parameters)!!
    :return: scale_factor dictionary
    """
    means = [np.mean(x) for x in scaler_list]
    return {'pos': {'features': param_list, 'factor': means}}


def read_in_file(path_to_file):
    if path_to_file.endswith("csv"):
        return pd.read_csv(path_to_file)

    elif path_to_file.endswith("fits"):
        fits_file = fits.open(path_to_file)
        data = fits_file[1].data
        df = pd.DataFrame(data.byteswap().newbyteorder())

        fits_file.close()

        return df


def save_output_summary(summary_str, file):
    """
    Save output summary (i.e. number of clusters found with each method and in consensus for a given sf of KNN value)
    to an ouptu file.

    :param summary_str: string of parameters to be saved
    :param file: filename + path
    :return:
    """

    # Write the output to the results file:
    df_row = pd.DataFrame(data=summary_str, index=[0])
    df_row.to_csv(file, mode="a", header=False)
