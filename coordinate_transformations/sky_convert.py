from astropy.coordinates import LSR, SkyCoord, Distance, Galactic, ICRS, CartesianRepresentation
import astropy.units as u
import pandas as pd


def to_SkyCoord(ra, dec, parallax, pmra, pmdec):
    # Transform to different coordinate system
    skycoord = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,  # 2D on sky postition
        distance=Distance(parallax=parallax * u.mas),  # distance in pc
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        frame="icrs",
    )
    return skycoord


def to_SkyCoord_use_distance(ra, dec, pmra, pmdec, dist):
    # Transform to different coordinate system
    skycoord = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,  # 2D on sky postition
        distance=dist * u.pc,  # distance in pc
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        frame="icrs",
    )
    return skycoord


def gal_vtan_lsr(skycoord):
    x = skycoord.galactic.cartesian.x.value
    y = skycoord.galactic.cartesian.y.value
    z = skycoord.galactic.cartesian.z.value
    # Transform to lsr
    pma_lsr = skycoord.transform_to(LSR()).pm_ra_cosdec.value
    pmd_lsr = skycoord.transform_to(LSR()).pm_dec.value
    v_a_lsr = 4.74047 * pma_lsr / (1000 / skycoord.distance.value)
    v_d_lsr = 4.74047 * pmd_lsr / (1000 / skycoord.distance.value)
    df = pd.DataFrame(
        {
            "X": x,
            "Y": y,
            "Z": z,
            "v_a_lsr": v_a_lsr,
            "v_d_lsr": v_d_lsr,
        }
    )
    return df


def transform_gal_cartesian_and_vtan_to_icrs_pm(X, Y, Z, v_a_lsr, v_d_lsr):
    # Create CartesianRepresentation for Galactic frame
    cart_repr = CartesianRepresentation(x=X * u.pc, y=Y * u.pc, z=Z * u.pc)

    # Create SkyCoord in Galactic frame using the representation
    pos_gal = SkyCoord(cart_repr, frame=Galactic)

    # The rest stays the same...
    dist_pc = pos_gal.distance.pc

    pm_l = (v_a_lsr * 1000) / (4.74047 * dist_pc) * u.mas / u.yr
    pm_b = (v_d_lsr * 1000) / (4.74047 * dist_pc) * u.mas / u.yr

    gal_with_pm = SkyCoord(
        l=pos_gal.l,
        b=pos_gal.b,
        distance=pos_gal.distance,
        pm_l_cosb=pm_l,
        pm_b=pm_b,
        frame=Galactic,
    )

    icrs = gal_with_pm.transform_to(ICRS())

    # Extract ra, dec, distance, pm_ra_cosdec, pm_dec
    ra = icrs.ra.deg
    dec = icrs.dec.deg
    parallax = (1000 / icrs.distance.pc)  # mas
    pmra = icrs.pm_ra_cosdec.to(u.mas/u.yr).value
    pmdec = icrs.pm_dec.to(u.mas/u.yr).value

    df = pd.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "parallax": parallax,
            "pmra": pmra,
            "pmdec": pmdec,
        }
    )

    return df


# See possible transformation functions as input for SigMA
def transform_sphere_to_cartesian(ra, dec, parallax, pmra, pmdec, dist=None, use_distance:bool = False):
    if not use_distance:
        skycoord = to_SkyCoord(ra, dec, parallax, pmra, pmdec)
    else:
        skycoord = to_SkyCoord_use_distance(ra, dec, pmra, pmdec, dist)

    df = gal_vtan_lsr(skycoord)
    return df


def idenity_transform(ra, dec, parallax, pmra, pmdec):
    df = pd.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "parallax": parallax,
            "pmra": pmra,
            "pmdec": pmdec,
        }
    )
    return df
