import pandas as pd
from coordinate_transformations.sky_convert import transform_sphere_to_cartesian

dr3_dict={
    'RAdeg': 'ra',
    'DEdeg': 'dec',
    'Plx': 'parallax',
    'pmRA': 'pmra',
    'pmDE': 'pmdec',
    'Pop': 'label',
    'e_RAdeg': 'ra_error',
    'e_DEdeg': 'dec_error',
    'e_Plx': 'parallax_error',
    'e_pmRA': 'pmra_error',
    'e_pmDE': 'pmdec_error',
    'RADEcor': 'ra_dec_corr',
    'RAPlxcor': 'ra_parallax_corr',
    'RApmRAcor': 'ra_pmra_corr',
    'RApmDEcor':'ra_pmdec_corr',
    'DEPlxcor': 'dec_parallax_corr',
    'DEpmRAcor': 'dec_pmra_corr',
    'DEpmDEcor': 'dec_pmdec_corr',
    'PlxpmRAcor': 'parallax_pmra_corr',
    'PlxpmDEcor':'parallax_pmdec_corr',
    'pmRApmDEcor': 'pmra_pmdec_corr',
    'RV': 'radial_velocity',
    'e_RV': 'radial_velocity_error'
}

# load the dataframe
# TODO: Change filepath
df = pd.read_csv(f'Vela_clusters_DR3.csv')

# rename columns according to SigMA colnames
df_renamed = df.rename(columns=dr3_dict)

# add distance
df_renamed["distance"] = 1000/df_renamed["parallax"]

# Calculate gal. cart. coordinates
df_xyz = transform_sphere_to_cartesian(ra=df_renamed.ra.to_numpy(),
                                       dec=df_renamed.dec.to_numpy(),
                                       parallax=df_renamed.parallax.to_numpy(),
                                       pmra=df_renamed.pmra.to_numpy(),
                                       pmdec=df_renamed.pmdec.to_numpy())
df_full = pd.concat([df_renamed, df_xyz], axis=1)

# TODO: Change output location
df_full.to_csv(f'Vela_clusters_DR3_preprocessed.csv')



