# DM battery memo (Phase 1)

Residual DM vs the product file-stem reference DM, per adapter, on all
24 co-detection products. One uniform config (windows: {'dsa': 5.0, 'chime': 4.0}, 256 ch, max 1024 samples). Entries: residual +- sigma [pc/cm3],
or the recorded reason when unconstrained. Descriptive only -- no
method is promoted here.

| product | arrival_regression | dmphase_variant_intree | dmpower_variant_intree | dm_phase_published | dm_power_published |
|---| --- | --- | --- | --- | --- |
| casey/chime | +0.000 +- 0.001 | +0.000 +- 0.000 | +0.020 +- 0.176 | -0.010 +- 0.000 | unconstrained (LinAlgError: SVD did not converge) |
| casey/dsa | unconstrained (only 0 sub-bands above S/N 4.0 (<3)) | +0.004 +- 0.033 | -0.017 +- 0.118 | -0.003 +- 0.031 | +0.007 +- 0.018 |
| chromatica/chime | +0.018 +- 0.002 | -0.030 +- 0.559 | +0.372 +- 0.568 | -0.012 +- 0.013 | +0.083 +- 0.020 |
| chromatica/dsa | +0.294 +- 0.047 | +0.337 +- 0.817 | +0.277 +- 1.486 | +0.493 +- 0.043 | -0.366 +- 1.478 |
| freya/chime | +0.014 +- 0.004 | +0.013 +- 0.009 | +0.088 +- 0.502 | +0.006 +- 0.010 | +0.030 +- 0.024 |
| freya/dsa | unconstrained (only 0 sub-bands above S/N 4.0 (<3)) | +0.090 +- 0.010 | +0.108 +- 0.680 | +0.067 +- 0.091 | +0.093 +- 0.018 |
| hamilton/chime | -0.028 +- 0.001 | -0.002 +- 0.005 | -0.005 +- 0.062 | -0.020 +- 0.000 | +0.001 +- 0.029 |
| hamilton/dsa | unconstrained (only 0 sub-bands above S/N 4.0 (<3)) | +0.015 +- 0.323 | +0.079 +- 0.999 | +0.150 +- 0.122 | -0.564 +- 1.918 |
| isha/chime | +0.125 +- 0.018 | -3.009 +- 0.755 | -0.887 +- 0.739 | +0.177 +- 0.029 | +0.247 +- 0.047 |
| isha/dsa | unconstrained (only 0 sub-bands above S/N 4.0 (<3)) | +0.089 +- 1.156 | -0.309 +- 0.963 | -0.670 +- 0.000 | +0.617 +- 3.187 |
| johndoeII/chime | -0.004 +- 0.007 | -0.021 +- 0.258 | +0.651 +- 0.624 | -0.012 +- 0.006 | +0.016 +- 0.025 |
| johndoeII/dsa | -0.021 +- 0.026 | -0.009 +- 0.041 | +0.001 +- 0.768 | +0.001 +- 0.033 | +0.008 +- 0.187 |
| mahi/chime | unconstrained (only 2 sub-bands above S/N 4.0 (<3)) | +1.021 +- 0.987 | -0.184 +- 0.716 | -3.280 +- 489.393 | +0.078 +- 0.050 |
| mahi/dsa | unconstrained (only 2 sub-bands above S/N 4.0 (<3)) | -0.055 +- 1.036 | -0.045 +- 0.972 | +0.147 +- 0.051 | +0.148 +- 2.159 |
| oran/chime | -0.326 +- 0.329 | +3.661 +- 0.316 | -0.662 +- 1.749 | -1.831 +- 18.522 | +0.257 +- 0.086 |
| oran/dsa | +0.027 +- 0.152 | -0.181 +- 0.805 | +0.008 +- 1.437 | +0.587 +- 0.254 | +0.031 +- 1.183 |
| phineas/chime | +0.185 +- 0.033 | +1.543 +- 0.977 | -0.729 +- 0.610 | +0.170 +- 0.031 | +0.163 +- 0.021 |
| phineas/dsa | unconstrained (only 0 sub-bands above S/N 4.0 (<3)) | -0.012 +- 0.014 | -0.004 +- 0.793 | -0.005 +- 0.031 | -0.031 +- 0.031 |
| whitney/chime | +0.004 +- 0.011 | +0.054 +- 0.372 | -0.007 +- 0.804 | +0.007 +- 0.003 | +0.003 +- 0.228 |
| whitney/dsa | +0.016 +- 0.015 | +0.047 +- 0.016 | +0.045 +- 0.864 | +0.004 +- 0.027 | +0.013 +- 0.035 |
| wilhelm/chime | -0.003 +- 0.013 | -0.184 +- 0.860 | +0.340 +- 0.808 | +0.221 +- 0.023 | +0.059 +- 0.071 |
| wilhelm/dsa | +0.054 +- 0.010 | +0.045 +- 0.032 | +0.026 +- 0.102 | +0.007 +- 0.049 | +0.010 +- 0.018 |
| zach/chime | +0.016 +- 0.006 | -0.007 +- 0.007 | +0.016 +- 0.343 | -0.008 +- 0.003 | +0.032 +- 0.020 |
| zach/dsa | -0.070 +- 0.023 | -0.011 +- 0.017 | -0.120 +- 1.955 | -0.019 +- 0.065 | +0.170 +- 2.321 |

## Cross-method scatter (std of constrained residuals per product)

- casey/chime: 0.011 (n=4)
- casey/dsa: 0.009 (n=4)
- chromatica/chime: 0.148 (n=5)
- chromatica/dsa: 0.297 (n=5)
- freya/chime: 0.030 (n=5)
- freya/dsa: 0.015 (n=4)
- hamilton/chime: 0.011 (n=5)
- hamilton/dsa: 0.284 (n=4)
- isha/chime: 1.242 (n=5)
- isha/dsa: 0.478 (n=4)
- johndoeII/chime: 0.263 (n=5)
- johndoeII/dsa: 0.010 (n=5)
- mahi/chime: 1.616 (n=4)
- mahi/dsa: 0.099 (n=4)
- oran/chime: 1.851 (n=5)
- oran/dsa: 0.259 (n=5)
- phineas/chime: 0.727 (n=5)
- phineas/dsa: 0.011 (n=4)
- whitney/chime: 0.022 (n=5)
- whitney/dsa: 0.018 (n=5)
- wilhelm/chime: 0.182 (n=5)
- wilhelm/dsa: 0.019 (n=5)
- zach/chime: 0.015 (n=5)
- zach/dsa: 0.098 (n=5)
