#!/usr/bin/env python
"""Demo script for FLITS batch processing."""

import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np

# Add scattering to path
sys.path.insert(0, str(Path(__file__).parent / "scattering"))

from scat_analysis.burstfit_pipeline import BurstPipeline, BurstDataset
from scat_analysis.config_utils import load_config
from flits.batch import ResultsDatabase, ScatteringResult

if __name__ == "__main__":
    print("="*60)
    print("DEMO: Running Scattering Analysis on casey/DSA")
    print("="*60)

    # Use existing config that has correct paths
    config_path = "scattering/configs/bursts/dsa/casey_dsa.yaml"
    config = load_config(config_path)

    print(f"\n📄 Config: {config_path}")
    print(f"   Data: {config.path}")
    print(f"   Telescope: {config.telescope.name}")
    print(f"   Downsampling: {config.pipeline.f_factor}x freq, {config.pipeline.t_factor}x time")

    # Create output directory
    output_dir = Path("demo_output/casey_dsa")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline with fewer steps for demo
    print(f"\n🚀 Creating pipeline...")
    pipe = BurstPipeline(
        inpath=config.path,
        outpath=output_dir,
        name="casey",
        dm_init=config.dm_init,
        telescope=config.telescope,
        sampler=config.sampler,
        f_factor=config.pipeline.f_factor,
        t_factor=config.pipeline.t_factor,
        steps=2000,  # Reduced for demo
        nproc=4,
    )

    # Create dataset to inspect data
    dataset = BurstDataset(
        inpath=pipe.inpath,
        outpath=pipe.outpath,
        name=pipe.name,
        telescope=config.telescope,
        sampler=config.sampler,
        f_factor=config.pipeline.f_factor,
        t_factor=config.pipeline.t_factor,
    )

    print(f"   Data shape: {dataset.data.shape}")
    print(f"   Time range: {dataset.time[0]:.2f} - {dataset.time[-1]:.2f} ms")
    print(f"   Freq range: {dataset.freq[0]:.3f} - {dataset.freq[-1]:.3f} GHz")

    # Quick sanity check
    data = dataset.data
    print(f"\n📈 Data statistics:")
    print(f"   Peak S/N location: t = {dataset.time[np.argmax(np.nansum(data, axis=0))]:.2f} ms")
    print(f"   Data range: [{np.nanmin(data):.4g}, {np.nanmax(data):.4g}]")

    print(f"\n⏳ Running MCMC (2000 steps for demo - normally use 10000)...")
    print("   This will take ~1-2 minutes...")

    results = pipe.run_full(
        model_scan=True,
        diagnostics=False,
        plot=True,
        show=False,
        save=True,
    )

    print(f"\n✅ Analysis complete!")
    print(f"   Best model: {results['best_key']}")
    print(f"   χ²/dof: {results['goodness_of_fit']['chi2_reduced']:.2f}")

    # Extract parameters
    best_p = results['best_params']
    print(f"\n📊 Best-fit parameters:")
    print(f"   τ_1GHz = {best_p.tau_1ghz:.4f} ms")
    print(f"   α = {best_p.alpha:.2f}")
    print(f"   t0 = {best_p.t0:.2f} ms")
    print(f"   ζ (width) = {best_p.zeta:.4f} ms")
    print(f"   γ (spectral) = {best_p.gamma:.2f}")

    # Store in database
    db = ResultsDatabase("flits_results.db")
    result = ScatteringResult.from_pipeline_results(
        burst_name="casey",
        telescope="dsa",
        results=results,
        config_path=config_path,
        data_path=str(config.path),
    )
    result.quality_flag = "good" if results['goodness_of_fit']['chi2_reduced'] < 2.0 else "marginal"
    db.add_scattering_result(result)
    db.close()

    print(f"\n💾 Results saved to database: flits_results.db")
    print(f"\n📁 Output files in: {output_dir}")
    for f in output_dir.glob("*.pdf"):
        print(f"   - {f.name}")
        
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)

