"""
Enhanced Gaia DR3 3D Stellar Visualization Script -- CSV Edition
Loads Gaia DR3 .csv data, filters safely, computes kNN densities, and builds a fast Plotly 3D view.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import SkyCoord, Galactocentric
from astropy import units as u
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = "/Users/laxman/projects/GitHub/Distant_SigMA/laxman_workspace/codes/ISM-FLOW-WS2/vela_unknown.csv"
OUTPUT_FILE = "/Users/laxman/projects/GitHub/Distant_SigMA/laxman_workspace/codes/ISM-FLOW-WS2/vela_unknown.html"

# Downsampling & browser speed
MAX_STARS = 25000
MAX_SCATTER_STARS = 20000

# Coordinates: keep HELIOCENTRIC by default; set to True for Galactocentric
USE_GALACTOCENTRIC = False

# Density / grid
DENSITY_K = 25
GRID_BINS = (60, 60, 40)
GAUSSIAN_SIGMA = 1.1

# Isosurfaces
ISOSURFACE_PERCENTILES = [99.8, 99.0, 95.0]
ISOSURFACE_OPACITIES  = [0.80, 0.40, 0.15]

# Visuals
COLORSCALE = 'ice_r'
POINT_SIZE_BASE = 2.0
POINT_SIZE_SCALE = 4.5
POINT_OPACITY = 0.78
DENSITY_CONTRAST_PERCENTILES = [2, 99.6]

BACKGROUND_COLOR = 'rgba(240, 247, 255, 1)'
AXIS_BACKGROUND = 'rgba(235, 245, 255, 0.2)'
GRID_COLOR = 'rgba(150, 170, 200, 0.13)'

# =============================================================================
def load_and_filter_data(file_path, max_stars=MAX_STARS):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, usecols=['ra', 'dec', 'parallax', 'pmra', 'pmdec'])
    df['distance'] = 1000.0 / df['parallax']
    if len(df) > max_stars:
        df = df.sample(n=max_stars, random_state=42)
    print(f"Loaded: {len(df):,} stars.")
    return df

# =============================================================================
# COORD CONVERSION
# =============================================================================
def convert_to_cartesian(df):
    print(f"Converting to {'Galactocentric' if USE_GALACTOCENTRIC else 'Galactic heliocentric'} Cartesian...")
    coords = SkyCoord(
        ra=df['ra'].values * u.deg,
        dec=df['dec'].values * u.deg,
        distance=df['distance'].values * u.pc,
        frame='icrs'
    )
    if USE_GALACTOCENTRIC:
        galcen = coords.transform_to(Galactocentric()).cartesian
        df['X'], df['Y'], df['Z'] = galcen.x.to(u.pc).value, galcen.y.to(u.pc).value, galcen.z.to(u.pc).value
    else:
        gal = coords.transform_to('galactic').cartesian
        df['X'], df['Y'], df['Z'] = gal.x.to(u.pc).value, gal.y.to(u.pc).value, gal.z.to(u.pc).value
    return df

# =============================================================================
# DENSITY & GRID
# =============================================================================
def calculate_stellar_density(df, k=DENSITY_K):
    print(f"Calculating {k}-NN density...")
    xyz = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k+1)
    rk = dists[:, -1]
    volume = (4.0/3.0) * np.pi * rk**3
    df['density'] = k / volume
    df['log_density'] = np.log10(df['density'])
    return df

def create_density_grid(df, bins=GRID_BINS, smooth_sigma=GAUSSIAN_SIGMA):
    x, y, z = df['X'].values, df['Y'].values, df['Z'].values
    H, edges = np.histogramdd([x, y, z], bins=bins, weights=df['density'].values)
    if smooth_sigma:
        H = gaussian_filter(H, sigma=smooth_sigma)
    xm = 0.5 * (edges[0][1:] + edges[0][:-1])
    ym = 0.5 * (edges[1][1:] + edges[1][:-1])
    zm = 0.5 * (edges[2][1:] + edges[2][:-1])
    X, Y, Z = np.meshgrid(xm, ym, zm, indexing='ij')
    return X, Y, Z, H

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_enhanced_visualization(df):
    fig = go.Figure()
    
    # Add isosurfaces
    try:
        X, Y, Z, H = create_density_grid(df)
        nonzero = H[H > 0]
        if nonzero.size > 0:
            thresholds = [np.percentile(nonzero, p) for p in ISOSURFACE_PERCENTILES]
            colors = pc.sample_colorscale(COLORSCALE, len(ISOSURFACE_PERCENTILES))
            flat, Xf, Yf, Zf = H.ravel(), X.ravel(), Y.ravel(), Z.ravel()
            for cutoff, opacity, color in zip(thresholds, ISOSURFACE_OPACITIES, colors):
                fig.add_trace(go.Isosurface(
                    x=Xf, y=Yf, z=Zf, value=flat,
                    isomin=cutoff, isomax=cutoff, surface_count=1,
                    opacity=opacity, colorscale=[[0, color], [1, color]],
                    showscale=False, caps=dict(x_show=False, y_show=False, z_show=False),
                    name=f'Iso ≥ {cutoff:.3g}'
                ))
    except Exception as e:
        print(f"[warn] Isosurface skipped: {e}")
    
    # Add scatter points
    scatter_df = df.sample(n=MAX_SCATTER_STARS, random_state=99) if len(df) > MAX_SCATTER_STARS else df
    lo, hi = np.nanpercentile(scatter_df['log_density'].values, DENSITY_CONTRAST_PERCENTILES)
    norm_density = np.clip((scatter_df['log_density'] - lo) / max(hi - lo, 1e-9), 0, 1)
    
    fig.add_trace(go.Scatter3d(
        x=scatter_df['X'], y=scatter_df['Y'], z=scatter_df['Z'],
        mode='markers',
        marker=dict(
            size=POINT_SIZE_BASE + POINT_SIZE_SCALE * norm_density,
            color=scatter_df['log_density'], colorscale=COLORSCALE,
            cmin=lo, cmax=hi, opacity=POINT_OPACITY,
            colorbar=dict(title="log₁₀(density)", thickness=16, len=0.67),
            line=dict(width=0)
        ),
        name='Stars'
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=("Vela Region: 3D Stellar Structure (Galactocentric)" if USE_GALACTOCENTRIC
                  else "Vela Region: 3D Stellar Structure (Heliocentric Galactic)"),
            x=0.5, font=dict(size=15)
        ),
        scene=dict(
            xaxis=dict(title='X (pc)', backgroundcolor=AXIS_BACKGROUND, gridcolor=GRID_COLOR, showbackground=True),
            yaxis=dict(title='Y (pc)', backgroundcolor=AXIS_BACKGROUND, gridcolor=GRID_COLOR, showbackground=True),
            zaxis=dict(title='Z (pc)', backgroundcolor=AXIS_BACKGROUND, gridcolor=GRID_COLOR, showbackground=True),
            aspectmode='cube', camera=dict(eye=dict(x=1.3, y=1.3, z=1.05)),
            bgcolor=BACKGROUND_COLOR
        ),
        autosize=True, paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        margin=dict(l=0, r=0, b=0, t=55), legend=dict(bgcolor='rgba(255,255,255,0.8)')
    )
    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*58)
    print("Gaia DR3 3D Stellar Density Visualization")
    print("="*58)
    df = load_and_filter_data(DATA_PATH)
    df = convert_to_cartesian(df)
    df = calculate_stellar_density(df, k=DENSITY_K)
    fig = create_enhanced_visualization(df)
    fig.write_html(OUTPUT_FILE, include_plotlyjs='cdn', full_html=True)
    print(f"Done! Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
