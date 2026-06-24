"""Pure CGM scattering and dispersion predictions from galaxy scalars."""

import math

import astropy.units as u
import numpy as np

from .config import COSMO


def _is_bad(x: float | None) -> bool:
    """Return True for missing, NaN, or non-finite scalar inputs."""
    if x is None:
        return True
    try:
        value = float(x)
    except (TypeError, ValueError):
        return True
    return math.isnan(value) or bool(np.isnan(value)) or not math.isfinite(value)


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a finite scalar to a closed interval."""
    return min(hi, max(lo, value))


def dm_halo_mnfw(
    m_halo_msun: float,
    z_gal: float,
    impact_kpc: float,
    f_hot: float = 0.75,
    y0: float = 2.0,
) -> float | None:
    """Return the observed mNFW hot-halo DM at projected impact parameter."""
    if any(_is_bad(x) for x in (m_halo_msun, z_gal, impact_kpc, f_hot, y0)):
        return None

    m200 = float(m_halo_msun)
    z_value = float(z_gal)
    b_kpc = float(impact_kpc)
    f_hot_value = float(f_hot)
    y0_value = float(y0)
    if m200 <= 0.0 or z_value < 0.0 or b_kpc < 0.0 or f_hot_value < 0.0 or y0_value < 0.0:
        return None

    from scipy.integrate import quad

    # Prochaska & Zheng 2019 MNRAS 485,648 set R200 from mass enclosing 200
    # times the redshift-dependent critical density, matching the mNFW halo
    # truncation used for CGM dispersion predictions in Macquart+2020.
    rho_crit = COSMO.critical_density(z_value).to(u.Msun / u.kpc**3).value
    if rho_crit <= 0.0 or not math.isfinite(rho_crit):
        return None
    r200_kpc = (3.0 * m200 / (4.0 * math.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    if r200_kpc <= 0.0 or not math.isfinite(r200_kpc):
        return None
    if b_kpc >= r200_kpc:
        return 0.0

    # Prochaska & Zheng 2019 MNRAS 485,648 use an NFW-like concentration for
    # halo gas; this bounded mass-concentration approximation keeps low/high
    # mass extrapolations numerically stable without changing the CGM trend.
    concentration = 7.7 * (m200 / 1.0e14) ** (-0.11)
    concentration = _clamp(concentration, 3.0, 15.0)
    rs_kpc = r200_kpc / concentration
    if rs_kpc <= 0.0 or not math.isfinite(rs_kpc):
        return None

    def shape_density(r_kpc: float) -> float:
        if r_kpc < 0.0:
            return 0.0
        y = r_kpc / rs_kpc
        # Prochaska & Zheng 2019 MNRAS 485,648 and Macquart+2020 Nature
        # 581,391 soften the NFW inner cusp by replacing y with y0+y, so the
        # column stays finite through the halo center while retaining an outer
        # NFW-like decline.
        return 1.0 / ((y0_value + y) * (1.0 + y) ** 2)

    def radial_mass_integrand(r_kpc: float) -> float:
        # The 4*pi*r^2 term is the spherical volume element needed to normalize
        # the mNFW profile to the hot baryon mass rather than an arbitrary rho0.
        return 4.0 * math.pi * r_kpc**2 * shape_density(r_kpc)

    norm_integral, _ = quad(
        radial_mass_integrand, 0.0, r200_kpc, epsabs=0.0, epsrel=1e-6, limit=200
    )
    if norm_integral <= 0.0 or not math.isfinite(norm_integral):
        return None

    # Macquart+2020 Nature 581,391 appendix anchors halo gas to the cosmic
    # baryon allotment; f_hot selects the ionized hot phase within that budget.
    baryon_fraction = float(COSMO.Ob0 / COSMO.Om0)
    gas_mass_msun = f_hot_value * baryon_fraction * m200
    if gas_mass_msun < 0.0 or not math.isfinite(gas_mass_msun):
        return None
    rho0_msun_kpc3 = gas_mass_msun / norm_integral

    # Prochaska & Zheng 2019 MNRAS 485,648 adopt mu_e~1.18 for fully ionized
    # primordial-composition CGM gas, converting gas mass density to electron
    # density through n_e=rho/(mu_e*m_p).
    mu_e = 1.18
    m_p_g = 1.67262192369e-24
    msun_per_kpc3_to_g_per_cm3 = (1.0 * u.Msun / u.kpc**3).to(u.g / u.cm**3).value
    rho0_g_cm3 = rho0_msun_kpc3 * msun_per_kpc3_to_g_per_cm3

    def ne_cm3(r_kpc: float) -> float:
        return rho0_g_cm3 * shape_density(r_kpc) / (mu_e * m_p_g)

    l_half_kpc = math.sqrt(max(r200_kpc**2 - b_kpc**2, 0.0))

    def los_integrand(l_kpc: float) -> float:
        r_kpc = math.hypot(b_kpc, l_kpc)
        return ne_cm3(r_kpc)

    column_ne_kpc_cm3, _ = quad(
        los_integrand, -l_half_kpc, l_half_kpc, epsabs=0.0, epsrel=1e-6, limit=200
    )
    if not math.isfinite(column_ne_kpc_cm3):
        return None

    # One kpc of path contributes 1000 pc to the conventional pc cm^-3 DM
    # units, and Macquart+2020 Nature 581,391 divides host/halo rest-frame DM
    # by (1+z) because dispersion is observed after cosmological dilation.
    dm_obs = column_ne_kpc_cm3 * 1000.0 / (1.0 + z_value)
    if not math.isfinite(dm_obs):
        return None
    return float(max(dm_obs, 0.0))


def dm_cool(
    dm_halo: float, cool_covering_fraction: float, mgii_wr: float | None = None
) -> float | None:
    """Return a cool-photoionized CGM DM component scaled by coverage."""
    if _is_bad(dm_halo) or _is_bad(cool_covering_fraction):
        return None

    dm_halo_value = float(dm_halo)
    fc = float(cool_covering_fraction)
    if dm_halo_value < 0.0 or fc < 0.0:
        return None
    fc = min(fc, 1.0)

    # Werk+2014 ApJ 792,8 and Prochaska+2017 motivate a sub-dominant but
    # nonzero T~10^4 K photoionized column; k=0.3 is an explicit prior that the
    # cool phase is comparable-to-lower than the hot halo electron column.
    k_eff = 0.3
    if not _is_bad(mgii_wr):
        # Lan & Mo 2018 ApJ 866,36 and Anand+2024 use MgII equivalent width as
        # a monotonic proxy for cool CGM column, so tanh gives a bounded EW
        # boost without letting saturated absorbers dominate the prior.
        k_eff *= 1.0 + 0.5 * math.tanh(float(mgii_wr) / 1.0)

    dm_value = dm_halo_value * fc * k_eff
    if not math.isfinite(dm_value):
        return None
    return float(max(dm_value, 0.0))


def f_tilde_prior(
    sfr_msun_yr: float,
    metallicity_12logOH: float | None = None,
    agn: bool = False,
) -> tuple[float, float, float]:
    """Return a monotonic prior bracket for the turbulent scattering factor."""
    sfr = 0.0 if _is_bad(sfr_msun_yr) else max(float(sfr_msun_yr), 0.0)

    # Ocker+2021 ApJ 911,102 and Cordes+2022 use F_tilde as a lumped
    # fluctuation factor; 0.1 is a quiescent-CGM order-of-magnitude floor in
    # the absence of a calibrated CGM F(SFR) relation.
    value = 0.1
    # Ocker+2021 ApJ 911,102 links stronger scattering to clumpier turbulent
    # ionized gas; log10(1+SFR) gives a transparent monotonic feedback boost
    # that is flat at zero SFR.
    value *= 1.0 + math.log10(1.0 + sfr)

    if not _is_bad(metallicity_12logOH):
        # Asplund+2009 ARA&A 47,481 gives solar 12+log(O/H)~8.7; higher
        # metallicity is used here as a mild dust/clumping proxy with bounded
        # leverage because no direct CGM F-metallicity calibration exists.
        metallicity_factor = 1.0 + 0.3 * (float(metallicity_12logOH) - 8.7)
        value *= _clamp(metallicity_factor, 0.5, 2.0)

    if agn:
        # Ocker+2021 ApJ 911,102 and Cordes+2022 associate stronger feedback
        # with enhanced turbulent density fluctuations, motivating a simple
        # factor-of-two AGN prior boost.
        value *= 2.0

    value = max(value, 1e-12)
    # Ocker+2021 ApJ 911,102 emphasizes dex-level uncertainty in extragalactic
    # scattering environments, so a factor-three bracket encodes roughly
    # 0.5 dex of prior width around the deterministic central value.
    lo = value / 3.0
    hi = value * 3.0
    return float(value), float(lo), float(hi)


def g_scatt(z_lens: float, z_frb: float) -> float:
    """Return intervening-screen geometric scattering leverage in Mpc."""
    if _is_bad(z_lens) or _is_bad(z_frb):
        return 0.0

    z_l = float(z_lens)
    z_s = float(z_frb)
    if z_l <= 0.0 or z_l >= z_s:
        return 0.0

    # Macquart & Koay 2013 ApJ 776,125 and Ocker+2021 ApJ 911,102 give the
    # thin-screen leverage D_L D_LS / D_S, so a screen at the source has zero
    # leverage as D_LS approaches zero.
    d_l = COSMO.angular_diameter_distance(z_l).to(u.Mpc).value
    d_s = COSMO.angular_diameter_distance(z_s).to(u.Mpc).value
    d_ls = COSMO.angular_diameter_distance_z1z2(z_l, z_s).to(u.Mpc).value
    if d_l <= 0.0 or d_s <= 0.0 or d_ls <= 0.0:
        return 0.0

    # Macquart & Koay 2013 ApJ 776,125 / Ocker+2021 ApJ 911,102: the intervening
    # thin-screen weight is the angular-diameter combination D_L D_LS / D_S (already
    # redshift-aware). The full observer-frame redshift dependence is the canonical
    # (1+z_l)^-3, applied ONCE in tau_scat_ms; we do NOT add a screen-dilation factor
    # here. The earlier extra /(1+z_l)^2 double-counted to (1+z_l)^-5 and
    # over-suppressed high-z screens, biasing the relative z-ranking of sightlines.
    leverage = d_l * d_ls / d_s
    if leverage <= 0.0 or not math.isfinite(leverage):
        return 0.0
    return float(leverage)


def tau_scat_ms(
    f_tilde: float,
    g_scatt_val: float,
    dm_l: float,
    z_lens: float,
    nu_ghz: float = 1.0,
) -> float | None:
    """Return pulse-broadening timescale from a lens-galaxy screen in ms."""
    if any(_is_bad(x) for x in (f_tilde, g_scatt_val, dm_l, z_lens, nu_ghz)):
        return None

    f_value = float(f_tilde)
    g_value = float(g_scatt_val)
    dm_value = float(dm_l)
    z_value = float(z_lens)
    nu_value = float(nu_ghz)
    if f_value < 0.0 or g_value < 0.0 or dm_value < 0.0 or z_value < 0.0 or nu_value <= 0.0:
        return None

    # Cordes & Lazio 2002 astro-ph/0207156, Cordes+2016 ApJ 832,113, and
    # Ocker+2021 ApJ 911,102 give tau proportional to SM*nu^-4 with
    # SM~F_tilde*DM^2 for a screen; A=1e-6 ms deliberately lumps the NE2001
    # SM-to-tau conversion and screen constants as an order-of-magnitude prior.
    a_ms = 1.0e-6
    # Macquart & Koay 2013 ApJ 776,125 convention for intervening screens uses
    # tau_obs=tau_rest*(1+z)^-3 after combining redshifted observing frequency
    # with cosmological time dilation, so the geometric weight is divided by
    # (1+z_lens)^3 here.
    tau = a_ms * f_value * g_value * dm_value**2 * nu_value ** (-4.0) / (1.0 + z_value) ** 3
    if not math.isfinite(tau):
        return None
    return float(max(tau, 0.0))


# Cool CGM enhances the turbulent fluctuation parameter F_tilde relative to the
# smooth hot halo: the cool phase shatters to ~pc cloudlets (McCourt+2018 MNRAS
# 473,5407), and such clumpy media can dominate radio scattering despite a
# sub-dominant electron column (Vedantham & Phinney 2019 MNRAS 483,971;
# Ocker+2021 ApJ 911,102). F_cool/F_hot is poorly constrained, so this is an
# explicit order-of-magnitude prior with a ~1 dex bracket.
COOL_CLUMP_BOOST = 10.0
COOL_CLUMP_BOOST_LO = 3.0
COOL_CLUMP_BOOST_HI = 30.0


def tau_scat_two_phase(
    f_tilde: float,
    g_scatt_val: float,
    dm_hot: float,
    dm_cool: float,
    z_lens: float,
    nu_ghz: float = 1.0,
    cool_clump_boost: float = COOL_CLUMP_BOOST,
) -> float | None:
    """Two-phase intervening-screen scattering: smooth hot halo + clumpy cool CGM.

    Scattering measures of independent screens add, and tau is proportional to
    SM, so the two phases combine linearly:

        tau = tau(F_tilde, dm_hot) + tau(F_tilde * cool_clump_boost, dm_cool)

    The cool phase carries a smaller electron column (``dm_cool``) but a much
    larger fluctuation parameter because it is clumpy on ~pc scales (McCourt+2018;
    Vedantham & Phinney 2019; Ocker+2021); ``cool_clump_boost`` = F_cool/F_hot is
    an explicit order-of-magnitude prior. A missing/NaN cool column degrades to
    the hot-only screen; if neither phase is usable, returns None.
    """
    hot = tau_scat_ms(f_tilde, g_scatt_val, dm_hot, z_lens, nu_ghz=nu_ghz)

    if _is_bad(f_tilde) or _is_bad(cool_clump_boost) or float(cool_clump_boost) < 0.0:
        f_tilde_cool: float | None = None
    else:
        f_tilde_cool = float(f_tilde) * float(cool_clump_boost)
    cool = tau_scat_ms(f_tilde_cool, g_scatt_val, dm_cool, z_lens, nu_ghz=nu_ghz)

    if hot is None and cool is None:
        return None
    total = (hot or 0.0) + (cool or 0.0)
    if not math.isfinite(total):
        return None
    return float(max(total, 0.0))


def scint_bandwidth_khz(tau_scat_ms: float) -> float | None:
    """Return scintillation decorrelation bandwidth from scattering time."""
    if _is_bad(tau_scat_ms):
        return None

    tau_ms = float(tau_scat_ms)
    if tau_ms <= 0.0:
        return None

    # Cordes & Rickett 1998 ApJ 507,846 give nu_dc=C1/(2*pi*tau) with
    # C1~1.16 for a uniform Kolmogorov medium, linking pulse broadening to
    # decorrelation bandwidth by Fourier reciprocity.
    bandwidth_hz = 1.16 / (2.0 * math.pi * (tau_ms * 1.0e-3))
    if bandwidth_hz <= 0.0 or not math.isfinite(bandwidth_hz):
        return None
    return float(bandwidth_hz / 1.0e3)


def predict_mgii_wr(impact_kpc: float, logmstar: float | None = None) -> float | None:
    """Return predicted MgII 2796 rest equivalent width in Angstrom."""
    if _is_bad(impact_kpc):
        return None

    r_kpc = float(impact_kpc)
    if r_kpc <= 0.0:
        return None

    r_break = 50.0
    w0 = 0.8
    if not _is_bad(logmstar):
        # Churchill+2013 ApJ 763,L42 and Anand+2021/2024 find stronger cool
        # absorption around more massive systems; this bounded 0.3-dex-per-dex
        # scaling keeps the Nielsen+2013 radial normalization recognizable.
        mass_factor = 10.0 ** (0.3 * (float(logmstar) - 10.5))
        w0 *= _clamp(mass_factor, 0.2, 3.0)

    # Nielsen+2013 ApJ 776,114 MAGIICAT and Anand+2021/2024 show MgII W_r
    # anti-correlates with projected distance; the broken power law allows a
    # steeper inner decline and shallower outer halo tail around 50 kpc.
    if r_kpc <= r_break:
        wr = w0 * (r_kpc / r_break) ** (-1.7)
    else:
        wr = w0 * (r_kpc / r_break) ** (-0.6)
    if wr <= 0.0 or not math.isfinite(wr):
        return None
    return float(wr)


def cool_covering_fraction(
    b_over_rvir: float,
    logmstar: float,
    is_star_forming: bool,
    phi_deg: float | None = None,
) -> tuple[float, float, float]:
    """Return a prior bracket for the cool CGM covering fraction."""
    b_scaled = 10.0 if _is_bad(b_over_rvir) else max(float(b_over_rvir), 0.0)

    # Lan & Mo 2018 ApJ 866,36 find cool-gas incidence declines with scaled
    # radius and is higher for star-forming galaxies; fc0=0.6/0.3 encodes that
    # population split while b_scale=0.5 gives an exponential halo falloff.
    fc0 = 0.6 if is_star_forming else 0.3
    if not _is_bad(logmstar):
        # Lan & Mo 2018 ApJ 866,36 show more cool absorption in more massive
        # halos on average, so this bounded tilt is deliberately mild.
        mass_tilt = 10.0 ** (0.15 * (float(logmstar) - 10.5))
        fc0 *= _clamp(mass_tilt, 0.5, 1.5)
    fc = fc0 * math.exp(-b_scaled / 0.5)

    if is_star_forming and not _is_bad(phi_deg):
        # Bordoloi+2011 ApJ 743,10 and Lan & Mo 2018 ApJ 866,36 motivate
        # stronger cool covering along the minor axis from outflow-fed gas, so
        # sin(phi) peaks at phi=90 deg and leaves the major axis unboosted.
        fc *= 1.0 + 0.5 * math.sin(math.radians(float(phi_deg)))

    fc = _clamp(fc, 0.0, 1.0)
    # Lan & Mo 2018 ApJ 866,36 highlight population and azimuthal scatter, so
    # the bracket widens around the central prior and gets extra upper room
    # when star-forming azimuth information is available.
    lo = _clamp(fc * 0.5, 0.0, 1.0)
    hi = _clamp(fc * 1.5 + (0.2 if (is_star_forming and phi_deg is not None) else 0.0), 0.0, 1.0)
    fc = _clamp(fc, lo, hi)
    return float(fc), float(lo), float(hi)


# --- Cluster ICM dispersion (isothermal beta-model) --------------------------
# Mohr+1999 ApJ 517,627 / Arnaud 2009: ICM beta ~ 0.6-0.7, core radius
# r_c ~ 0.1-0.2 R500. Cavaliere & Fusco-Femiano 1976 A&A 49,137 set the form.
CLUSTER_BETA = 0.65
CLUSTER_RC_OVER_R500 = 0.15
# Eckert+2019 A&A 621,A40 (X-COP) / Vikhlinin+2006: the hot-gas mass fraction
# within R500 spans ~0.10-0.13 and is lower at the smaller radii where most of the
# projected column is built. We adopt the lower-middle 0.11 (the 0.13 high end
# inflates the DM); even so the isothermal beta-model over-predicts the outskirts,
# so dm_cluster_beta_model returns an UPPER BOUND (see its docstring).
CLUSTER_F_GAS_500 = 0.11
# Fully ionized ICM mean molecular weight per electron (primordial-ish).
CLUSTER_MU_E = 1.17
# R200/R500 = (M200/M500 * 500/200)^(1/3) = (1.3*2.5)^(1/3) = 1.48, consistent with
# config.CLUSTER_M500_TO_M200 = 1.3 (typical c500 ~ 1.5). The earlier 1.54 assumed a
# different concentration and was inconsistent with the 1.3 mass ratio used for R200.
CLUSTER_R200_OVER_R500 = 1.48


def r_delta_kpc(m_delta_msun: float, z: float, delta: float) -> float:
    """Spherical-overdensity radius R_Delta (kpc): M = (4/3) pi Delta rho_c(z) R^3.

    For Delta=200 this equals the R200 used in get_rvir_and_rs
    ((G M/(100 H^2))^(1/3)); written via the critical density to match the R200
    derivation already in dm_halo_mnfw.
    """
    if any(_is_bad(x) for x in (m_delta_msun, z)) or float(m_delta_msun) <= 0.0:
        return float("nan")
    rho_crit = COSMO.critical_density(float(z)).to(u.Msun / u.kpc**3).value
    if rho_crit <= 0.0 or not math.isfinite(rho_crit):
        return float("nan")
    radius = (3.0 * float(m_delta_msun) / (4.0 * math.pi * float(delta) * rho_crit)) ** (1.0 / 3.0)
    return float(radius)


def _beta_mass_integral_kpc3(beta: float, rc_kpc: float, r_max_kpc: float) -> float:
    """Shape-only gas-mass integral int_0^Rmax 4 pi r^2 [1+(r/rc)^2]^(-3beta/2) dr (kpc^3)."""
    from scipy.integrate import quad

    integrand = lambda r: r * r * (1.0 + (r / rc_kpc) ** 2) ** (-1.5 * beta)  # noqa: E731
    value, _ = quad(integrand, 0.0, r_max_kpc, epsabs=0.0, epsrel=1e-7, limit=200)
    return 4.0 * math.pi * value


def _beta_ne0_cm3(
    m500_msun: float,
    z: float,
    r500_kpc: float,
    beta: float,
    rc_kpc: float,
    f_gas: float = CLUSTER_F_GAS_500,
) -> float:
    """Central electron density (cm^-3) such that M_gas(<R500) = f_gas * M500.

    M_gas = mu_e m_p n_e0 * int 4 pi r^2 shape(r) dr, so n_e0 follows from the
    shape integral normalized to the catalog gas mass.
    """
    shape_int_kpc3 = _beta_mass_integral_kpc3(beta, rc_kpc, r500_kpc)
    if shape_int_kpc3 <= 0.0 or not math.isfinite(shape_int_kpc3):
        return float("nan")
    m_gas_g = float(f_gas) * float(m500_msun) * u.Msun.to(u.g)
    mu_e_mp_g = CLUSTER_MU_E * 1.67262192369e-24
    kpc3_to_cm3 = (1.0 * u.kpc**3).to(u.cm**3).value
    return m_gas_g / (mu_e_mp_g * shape_int_kpc3 * kpc3_to_cm3)


def dm_cluster_beta_model(
    m500_msun: float,
    z: float,
    impact_kpc: float,
    r500_kpc: float | None = None,
    beta: float = CLUSTER_BETA,
    rc_over_r500: float = CLUSTER_RC_OVER_R500,
    f_gas: float = CLUSTER_F_GAS_500,
    r_trunc_factor: float = CLUSTER_R200_OVER_R500,
) -> float:
    """Observer-frame ICM dispersion measure (pc cm^-3) through a beta-model cluster.

    Isothermal beta-model n_e(r) = n_e0 [1+(r/r_c)^2]^(-3beta/2) (Cavaliere &
    Fusco-Femiano 1976), normalized so the hot gas within R500 is f_gas * M500
    (Vikhlinin+2006). The column is projected at impact parameter b, truncated at
    r_trunc_factor * R500, and divided by (1+z) for the observer frame
    (Macquart+2020 convention). Returns 0.0 for b beyond truncation or bad inputs.

    UPPER BOUND: the isothermal beta-model keeps the density slope fixed to large
    radius, but real clusters steepen beyond ~R500, so this over-predicts the
    projected column by ~1.5-2x relative to the resolved anchor -- Lee+2023 give
    ~300 and ~110 pc cm^-3 at b ~ 0.4 and 0.8 R200 for the FRB 20190520B foreground
    clusters, where this model returns ~1.7-1.9x more. Treat the value as a
    conservative ceiling on the cluster DM, not a point estimate; the upgrade path
    is a Vikhlinin+2006 steepening term or a per-cluster X-ray n_e(r) profile.
    """
    if any(_is_bad(x) for x in (m500_msun, z, impact_kpc)) or float(m500_msun) <= 0.0:
        return 0.0
    if r500_kpc is None or _is_bad(r500_kpc):
        r500_kpc = r_delta_kpc(m500_msun, z, 500)
    if _is_bad(r500_kpc) or float(r500_kpc) <= 0.0:
        return 0.0
    r500_kpc = float(r500_kpc)
    rc = float(rc_over_r500) * r500_kpc
    r_trunc = float(r_trunc_factor) * r500_kpc
    b = float(impact_kpc)
    if b >= r_trunc:
        return 0.0
    ne0 = _beta_ne0_cm3(m500_msun, z, r500_kpc, beta, rc, f_gas=f_gas)
    if _is_bad(ne0):
        return 0.0

    from scipy.integrate import quad

    l_half = math.sqrt(max(r_trunc**2 - b**2, 0.0))

    def los(l_kpc: float) -> float:
        r = math.hypot(b, l_kpc)
        return ne0 * (1.0 + (r / rc) ** 2) ** (-1.5 * beta)

    column_kpc_cm3, _ = quad(los, -l_half, l_half, epsabs=0.0, epsrel=1e-6, limit=200)
    if not math.isfinite(column_kpc_cm3):
        return 0.0
    # 1 kpc of path contributes 1000 pc to the pc cm^-3 DM unit; /(1+z) for the
    # observer frame (Macquart+2020).
    dm_obs = column_kpc_cm3 * 1000.0 / (1.0 + float(z))
    return float(max(dm_obs, 0.0))
