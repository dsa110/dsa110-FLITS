
import numpy as np
from scattering.scat_analysis.burstfit import FRBParams, FRBModel, analytic_gaussian_exp_convolution

def test_frbparams_initialization_and_aliases():
    """Test FRBParams initialization and property aliases."""
    params = FRBParams(c0=10.0, t0=5.0, gamma=-1.5, zeta=0.1, tau_1ghz=0.2, alpha=4.0)
    
    assert params.c0 == 10.0
    assert params.t0 == 5.0
    assert params.gamma == -1.5
    assert params.zeta == 0.1
    assert params.tau_1ghz == 0.2
    assert params.alpha == 4.0
    
    # Test Aliases
    assert params.amplitude == 10.0
    assert params.width == 0.1
    assert params.tau_alpha == 4.0

def test_frbparams_defaults():
    """Test default values for optional parameters."""
    params = FRBParams(c0=10.0, t0=5.0, gamma=-1.5)
    assert params.zeta == 0.0
    assert params.tau_1ghz == 0.0
    assert params.alpha == 4.4
    assert params.delta_dm == 0.0

def test_frbparams_sequence_conversion():
    """Test to_sequence and from_sequence methods."""
    params = FRBParams(c0=10.0, t0=5.0, gamma=-1.5, zeta=0.1, tau_1ghz=0.2, alpha=4.0, delta_dm=0.01)
    
    # Test M3 sequence
    seq_m3 = params.to_sequence("M3")
    assert len(seq_m3) == 7
    assert np.allclose(seq_m3, [10.0, 5.0, -1.5, 0.1, 0.2, 4.0, 0.01])
    
    # Test reconstruction
    params_new = FRBParams.from_sequence(seq_m3, "M3")
    assert params_new.c0 == params.c0
    assert params_new.t0 == params.t0
    assert params_new.alpha == params.alpha

    # Test M0 sequence (subset)
    seq_m0 = params.to_sequence("M0")
    assert len(seq_m0) == 3
    assert np.allclose(seq_m0, [10.0, 5.0, -1.5])

def test_analytic_convolution_gaussian_limit():
    """Test analytic convolution reduces to Gaussian when tau -> 0."""
    t = np.linspace(-5, 5, 100)
    mu = 0.0
    sig = 1.0
    tau = np.array([1e-10]) # Very small tau
    
    res = analytic_gaussian_exp_convolution(t, mu, sig, tau)
    
    # Standard Gaussian
    expected = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((t - mu) / sig)**2)
    
    # Should be very close
    np.testing.assert_allclose(res.squeeze(), expected, rtol=1e-5, atol=1e-8)

def test_frbmodel_initialization():
    """Test FRBModel initialization."""
    time = np.linspace(0, 10, 100)
    freq = np.linspace(1.2, 1.5, 32)
    
    model = FRBModel(time, freq)
    assert model.time.shape == (100,)
    assert model.freq.shape == (32,)
    assert model.dt > 0

def test_frbmodel_simulate_m0():
    """Test basic Gaussian model (M0) simulation."""
    time = np.linspace(0, 10, 100)
    freq = np.array([1.4])
    model = FRBModel(time, freq, dm_init=0.0)
    
    params = FRBParams(c0=1.0, t0=5.0, gamma=0.0, zeta=0.5)
    
    # Call model (Use M1 so zeta is used)
    spec = model(params, "M1")
    
    # Check peak position
    profile = spec[0]
    peak_idx = np.argmax(profile)
    assert np.abs(time[peak_idx] - 5.0) < (time[1] - time[0])

def test_frbmodel_dispersion():
    """Test that dispersion delay is applied correctly."""
    time = np.linspace(0, 100, 1000)
    freq = np.array([1.5, 1.2]) # High to low freq
    model = FRBModel(time, freq, dm_init=0.0) # dm_init=0, we add delta_dm
    
    # Add significant DM
    delta_dm = 50.0
    params = FRBParams(c0=1.0, t0=10.0, gamma=0.0, zeta=0.5, delta_dm=delta_dm)
    
    spec = model(params, "M1")
    
    t_peak_high = time[np.argmax(spec[0])]
    t_peak_low = time[np.argmax(spec[1])]
    
    # Delay propto freq^-2
    # delay = 4.15 * DM * (nu_low^-2 - nu_high^-2)
    # Check if low freq arrives later
    assert t_peak_low > t_peak_high

def test_frbmodel_smearing():
    """Test that smearing increases width."""
    time = np.linspace(0, 20, 200)
    freq = np.array([1.0])
    
    # Case 1: No DM, small width
    model1 = FRBModel(time, freq, dm_init=0.0)
    params1 = FRBParams(c0=1.0, t0=10.0, gamma=0.0, zeta=0.1)
    spec1 = model1(params1, "M1")
    
    # Case 2: High DM, same intrinsic width -> larger effective width due to smearing
    model2 = FRBModel(time, freq, dm_init=1000.0, df_MHz=10.0) # Large DM and channel width
    spec2 = model2(params1, "M1")
    
    # Estimate FWHM or just peak height (height should decrease as width increases for fixed area)
    # Note: The model output is normalized?
    # Looking at code: gauss / safe_gauss_sum * amp
    # So area is conserved per channel if we sum over time.
    # If width increases, peak should decrease.
    
    max1 = np.max(spec1)
    max2 = np.max(spec2)
    
    assert max2 < max1
