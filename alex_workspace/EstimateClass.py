import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance
from itertools import combinations


from NoiseRemoval.BulkVelocityClassic import ClassicBV
from miscellaneous.covariance_trafo_sky2gal import transform_covariance_shper2gal


class EstimatorClass:
    """Class to estimate the Mahalanobis distance between two clusters of data."""
    def __init__(self,data):
        self.data = data
        self.cbve = ClassicBV(self.data)
        self.file_path = ""

    def _load_data(self):
        """Load and preprocess the data from the CSV file."""
        df = pd.read_csv(self.file_path)

        cols2keep = [
            'source_id_1', 'ra_1', 'dec_1', 'parallax_1', 'pmra_1', 'pmdec_1',  # 'radial_velocity_1',
            'RV', 'RV_error',
            'pmra_error_1', 'pmdec_error_1', 'parallax_error_1',  # 'radial_velocity_error_1',
            'X', 'Y', 'Z', 'U', 'V', 'W',
            'cluster_label',
        ]

        data = df[cols2keep].rename(columns={col: col.removesuffix('_1') for col in cols2keep if col.endswith('_1')})
        data = data.rename(columns={'RV': 'radial_velocity', 'RV_error': 'radial_velocity_error'})
        return data



    def estimate_maha_distance(self, indices_1, indices_2):
        """In this version the cbve object is directly passed (which is instantiated outside of the function)."""
        # Input check
        cbve = self.cbve
        if indices_1.dtype != bool:
            bool_arr_1 = np.isin(np.arange(cbve.data.shape[0]), indices_1)
            bool_arr_2 = np.isin(np.arange(cbve.data.shape[0]), indices_2)
        else:
            bool_arr_1 = indices_1
            bool_arr_2 = indices_2

        data_subset = cbve.data.loc[~cbve.rv_isnan & (bool_arr_1 | bool_arr_2)]
        C_vel = np.diag([data_subset.pmra_error.mean(), data_subset.pmdec_error.mean(),
                         data_subset.radial_velocity_error.mean()]) ** 2
        # This creates a copy of the data --> not efficient and better to pass the object created to this function instead of the data (see below)
        # Estimate mean and covariance of the two clusters
        mu_1, cov_1 = cbve.estimate_normal_params(cluster_subset=bool_arr_1, method='BFGS')
        mu_2, cov_2 = cbve.estimate_normal_params(cluster_subset=bool_arr_2, method='BFGS')
        # Compute local covariance matrix in UVW space
        ra, dec, plx = cbve.data.loc[(bool_arr_1 | bool_arr_2), ['ra', 'dec', 'parallax']].mean().values.reshape(3, 1)
        C_uvw = transform_covariance_shper2gal(ra, dec, plx, C_vel.reshape(1, 3, 3))[0]
        # Compute Mahalanobis distance
        min_maha = np.min([
            distance.mahalanobis(mu_1, mu_2, np.linalg.inv(cov_1 + C_uvw)),
            distance.mahalanobis(mu_1, mu_2, np.linalg.inv(cov_2 + C_uvw))
        ])
        return min_maha
