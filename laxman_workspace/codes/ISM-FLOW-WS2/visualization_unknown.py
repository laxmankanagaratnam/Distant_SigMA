"""
Enhanced Gaia DR3 3D Stellar Visualization Script -- CSV Edition
Loads Gaia DR3 .csv data, filters safely, computes kNN densities, and builds a fast Plotly 3D view.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ------------------------- Optional deps -------------------------
try:
    from astropy.coordinates import SkyCoord, Galactocentric
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except Exception:
    ASTROPY_AVAILABLE = False

try:
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = "vela_unknown.csv"
OUTPUT_FILE = "vela_unknown.html"

# Downsampling & browser speed
MAX_STARS = 25000
MAX_SCATTER_STARS = 20000

# Physical cuts (pc)
DISTANCE_MIN = 10
DISTANCE_MAX = 700
PARALLAX_ERROR_THRESHOLD = 0.2   # relative error threshold (mas/mas)

# Coordinates: keep HELIOCENTRIC by default; set to True for Galactocentric
USE_GALACTOCENTRIC = False

# Density / grid
DENSITY_K = 25
GRID_BINS = (60, 60, 40)         # moderate grid (small HTML)
GAUSSIAN_SIGMA = 1.1

ENHANCED_ISOSURFACES = True
ISOSURFACE_PERCENTILES = [99.99, 99.8, 99.5, 99.0, 97.5, 95.0, 90.0]
ISOSURFACE_OPACITIES  = [0.98,   0.70,  0.50, 0.25,  0.10,  0.05, 0.018]

# Visuals
COLORSCALE = 'ice'
POINT_SIZE_BASE = 1.4
POINT_SIZE_SCALE = 3.5
POINT_OPACITY = 0.78
DENSITY_CONTRAST_PERCENTILES = [2, 99.6]

BACKGROUND_COLOR = 'rgba(240, 247, 255, 1)'
AXIS_BACKGROUND = 'rgba(235, 245, 255, 0.2)'
GRID_COLOR = 'rgba(150, 170, 200, 0.13)'

# =============================================================================
# UTILS
# =============================================================================
def pick_column(df, key_exact, fallback=None, contains=None):
    cols = {c.lower(): c for c in df.columns}
    if key_exact and key_exact.lower() in cols:
        return cols[key_exact.lower()]
    if contains:
        for c in df.columns:
            if contains.lower() in c.lower():
                return c
    if fallback is not None:
        return fallback
    raise KeyError(f"Column not found: exact='{key_exact}', contains='{contains}'")

# =============================================================================
# LOAD + FILTER
# =============================================================================
def load_and_filter_data(file_path, max_stars=MAX_STARS):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Column mapping (robust)
    ra_col   = pick_column(df, 'ra',  fallback=list(df.columns)[0])
    dec_col  = pick_column(df, 'dec', fallback=list(df.columns)[1])
    plx_col  = pick_column(df, 'parallax', contains='parallax')
    plxe_col = pick_column(df, 'parallax_error', contains='parallax_error')

    df = df[[ra_col, dec_col, plx_col, plxe_col]].rename(
        columns={ra_col: 'ra', dec_col: 'dec', plx_col: 'parallax', plxe_col: 'parallax_error'}
    )

    # Gaia parallax & error are in mas. Distance (pc) = 1000 / parallax(mas)
    # Ensure that plx>0 and finite:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['parallax', 'parallax_error', 'ra', 'dec'])
    df = df[df['parallax'] > 0]

    df['distance'] = 1000.0 / df['parallax']  # pc

    # Physical filters in pc + relative error cut
    df = df[(df['distance'] >= DISTANCE_MIN) & (df['distance'] <= DISTANCE_MAX)]
    df = df[(df['parallax_error'] / df['parallax']) < PARALLAX_ERROR_THRESHOLD]

    # Downsample for performance
    if len(df) > max_stars:
        df = df.sample(n=max_stars, random_state=42)

    df = df[np.isfinite(df['distance']) & (df['distance'] > 0)]
    n = len(df)
    if n == 0:
        raise RuntimeError(
            "Nach dem Filtern sind 0 Sterne übrig.\n"
            "-> Erhöhe DISTANCE_MAX (z.B. 1200), lockere PARALLAX_ERROR_THRESHOLD (z.B. 0.3), "
            "oder prüfe die Spaltennamen/Einheiten im VOTable."
        )
    print(f"Loaded and filtered: {n:,} stars.")
    return df

# =============================================================================
# COORD CONVERSION
# =============================================================================
def convert_to_cartesian(df):
    if not ASTROPY_AVAILABLE:
        raise RuntimeError("Astropy is required for coordinate transforms.")
    print(f"Converting to {'Galactocentric' if USE_GALACTOCENTRIC else 'Galactic heliocentric'} Cartesian...")

    coords = SkyCoord(
        ra=df['ra'].values * u.deg,
        dec=df['dec'].values * u.deg,
        distance=df['distance'].values * u.pc,
        frame='icrs'
    )

    if USE_GALACTOCENTRIC:
        # Astropy macht die korrekte Frame-Transformation
        galcen = coords.transform_to(Galactocentric())
        cart = galcen.cartesian
        df['X'], df['Y'], df['Z'] = cart.x.to(u.pc).value, cart.y.to(u.pc).value, cart.z.to(u.pc).value
    else:
        # Heliozentrisch im galaktischen Frame (kein manuelles Abziehen der Sonnenposition!)
        gal = coords.transform_to('galactic')
        cart = gal.cartesian
        df['X'], df['Y'], df['Z'] = cart.x.to(u.pc).value, cart.y.to(u.pc).value, cart.z.to(u.pc).value

    return df

# =============================================================================
# DENSITY & GRID
# =============================================================================
def calculate_stellar_density(df, k=DENSITY_K):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy required for kNN density (cKDTree).")
    if len(df) <= k:
        raise RuntimeError(f"Zu wenige Punkte ({len(df)}) für k-NN mit k={k}. Reduziere DENSITY_K oder lockere Filter.")
    print(f"Calculating {k}-NN density...")

    xyz = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k+1)
    rk = dists[:, -1]  # Entfernung zum k-ten Nachbarn
    volume = (4.0/3.0) * np.pi * rk**3
    dens = k / volume
    df['density'] = dens
    df['log_density'] = np.log10(dens)
    return df

def create_density_grid(df, bins=GRID_BINS, smooth_sigma=GAUSSIAN_SIGMA):
    x, y, z = df['X'].values, df['Y'].values, df['Z'].values
    H, edges = np.histogramdd([x, y, z], bins=bins, weights=df['density'].values)
    if smooth_sigma and SCIPY_AVAILABLE:
        H = gaussian_filter(H, sigma=smooth_sigma)
    xm = 0.5*(edges[0][1:] + edges[0][:-1])
    ym = 0.5*(edges[1][1:] + edges[1][:-1])
    zm = 0.5*(edges[2][1:] + edges[2][:-1])
    X, Y, Z = np.meshgrid(xm, ym, zm, indexing='ij')
    return X, Y, Z, H

def sample_colorscale_enhanced(colorscale, n):
    return pc.sample_colorscale(colorscale, n)

def add_enhanced_isosurfaces(fig, X, Y, Z, H, percentiles=ISOSURFACE_PERCENTILES,
                             opacities=ISOSURFACE_OPACITIES, colorscale=COLORSCALE):
    if not ENHANCED_ISOSURFACES:
        return fig
    nonzero = H[H > 0]
    if nonzero.size == 0:
        # Kein Signal -> Isoflächen überspringen
        return fig
    thresholds = [np.percentile(nonzero, p) for p in percentiles]
    colors = sample_colorscale_enhanced(colorscale, len(percentiles))
    flat = H.ravel()
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    for cutoff, opacity, color in zip(thresholds, opacities, colors):
        fig.add_trace(go.Isosurface(
            x=Xf, y=Yf, z=Zf, value=flat,
            isomin=cutoff, isomax=cutoff, surface_count=1,
            opacity=opacity,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=f'Iso ≥ {cutoff:.3g}'
        ))
    return fig

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_enhanced_visualization(df):
    fig = go.Figure()

    # Grid & Isoflächen
    try:
        X, Y, Z, H = create_density_grid(df)
        fig = add_enhanced_isosurfaces(fig, X, Y, Z, H)
    except Exception as e:
        print(f"[warn] Isosurface skipped: {e}")

    # Scatter (limitiert für Speed)
    scatter_df = df.sample(n=MAX_SCATTER_STARS, random_state=99) if len(df) > MAX_SCATTER_STARS else df
    if 'log_density' in scatter_df:
        lo, hi = np.nanpercentile(scatter_df['log_density'].values, DENSITY_CONTRAST_PERCENTILES)
        norm_density = np.clip((scatter_df['log_density'] - lo) / max(hi - lo, 1e-9), 0, 1)
        cvals = scatter_df['log_density']
        cmin, cmax = lo, hi
    else:
        # Fallback falls Dichte nicht berechnet
        norm_density = np.zeros(len(scatter_df))
        cvals = None
        cmin = cmax = None

    fig.add_trace(go.Scatter3d(
        x=scatter_df['X'], y=scatter_df['Y'], z=scatter_df['Z'],
        mode='markers',
        marker=dict(
            size=POINT_SIZE_BASE + POINT_SIZE_SCALE*norm_density,
            color=cvals,
            colorscale=COLORSCALE,
            cmin=cmin, cmax=cmax,
            opacity=POINT_OPACITY,
            colorbar=dict(title="log₁₀(density)", thickness=16, len=0.67) if cvals is not None else None,
            line=dict(width=0),
        ),
        name='Stars'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=("Vela-Puppis Region (Unknown): 3D Stellar Structure (Galactocentric)" if USE_GALACTOCENTRIC
                  else "Vela-Puppis Region (Unknown): 3D Stellar Structure (Heliocentric Galactic)"),
            x=0.5, font=dict(size=15)
        ),
        scene=dict(
            xaxis=dict(title='X (pc)', backgroundcolor=AXIS_BACKGROUND, gridcolor=GRID_COLOR, showbackground=True),
            yaxis=dict(title='Y (pc)', backgroundcolor=AXIS_BACKGROUND, gridcolor=GRID_COLOR, showbackground=True),
            zaxis=dict(title='Z (pc)', backgroundcolor=AXIS_BACKGROUND, gridcolor=GRID_COLOR, showbackground=True),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.05)),
            bgcolor=BACKGROUND_COLOR
        ),
        autosize=True,
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        margin=dict(l=0, r=0, b=0, t=55),
        legend=dict(bgcolor='rgba(255,255,255,0.8)')
    )
    return fig

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*58)
    print("Enhanced Gaia DR3 3D Stellar Density Visualization (CSV version)")
    print("="*58)
    df = load_and_filter_data(DATA_PATH)
    df = convert_to_cartesian(df)
    df = calculate_stellar_density(df, k=DENSITY_K)
    fig = create_enhanced_visualization(df)
    print("\nSaving to:", OUTPUT_FILE)
    fig.write_html(OUTPUT_FILE, include_plotlyjs='cdn', full_html=True)
    print("Done! Open", OUTPUT_FILE, "in your browser.")

if __name__ == "__main__":
    main()
