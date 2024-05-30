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
    def __init__(self, file_path = r'C:\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\Git\SigMA_Alex_modifications\alex_workspace\3D_plotting\3D_plotting\Region_dataframes\Region_0.0_sf_200_grouped_solutions.csv'
):
        self.file_path = file_path
        self.data = self._load_data()
        self.cbve = ClassicBV(self.data)

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
        """Estimate the Mahalanobis distance between two clusters of data."""
        if indices_1.dtype == bool:
            idx_1 = np.where(indices_1)[0]
            idx_2 = np.where(indices_2)[0]
        else:
            idx_1 = indices_1
            idx_2 = indices_2

        data_subset = self.data.loc[np.union1d(idx_1, idx_2)]
        data_subset = data_subset.loc[~data_subset.radial_velocity_error.isna()]
        C_vel = np.diag(
            [data_subset.pmra_error.mean(), data_subset.pmdec_error.mean(),
             data_subset.radial_velocity_error.mean()]) ** 2

        # Estimate mean and covariance of the two clusters
        mu_1, cov_1 = self.cbve.estimate_normal_params(cluster_subset=idx_1, method='BFGS')
        mu_2, cov_2 = self.cbve.estimate_normal_params(cluster_subset=idx_2, method='BFGS')

        # Compute local covariance matrix in UVW space
        ra, dec, plx = self.data.loc[np.union1d(idx_1, idx_2), ['ra', 'dec', 'parallax']].median().values.reshape(3, 1)
        C_uvw = transform_covariance_shper2gal(ra, dec, plx, C_vel.reshape(1, 3, 3))[0]

        min_maha = np.min([
            distance.mahalanobis(mu_1, mu_2, np.linalg.inv(cov_1 + C_uvw)),
            distance.mahalanobis(mu_1, mu_2, np.linalg.inv(cov_2 + C_uvw))
        ])
        return min_maha


