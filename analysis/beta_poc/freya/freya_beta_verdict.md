# freya beta co-model verdict (dsa110-FLITS #106)

**Grade: PASS** - provisional-citable: **True**

- beta = 3.6837 +0.0128/-0.0136 (un-railed: 50 sigma / 25 sigma from the [3.0, 4.0) edges)
- derived alpha = 4.3757 +0.0193/-0.0179 (thin-screen closure; NOT independently fit)
- tau_1GHz = 0.11439 ms +0.00111/-0.00113
- lnZ = -24144.12 +/- 0.55
- PPC chi2/dof: CHIME 1.18, DSA 1.03 (Level-2 PASS band [0.3, 1.5])
- Route A validation: PASS
- x_zeta-beta posterior correlation r = +0.011 (benign)
- A-vs-B (#105): **agree** on all physics params
- Exp-era comparison (#100 comparator): overall **agree**; |delta alpha| = 0.020 <= 0.1 -> within the manuscript wording-only claim band. (exp-era value is the deprecated free-alpha+exponential-PBF fit's suggestion -- the hypothesis under test, not citable truth.)

## Caveat

SENSITIVITY-REGIME CAVEAT (binding): at the fitted candidate the CHIME window captures 2.07 e-folds (87.4%) of the power-law PBF tail, below the 3.0 preflight threshold. Both raw captures are 81.9 ms burst-centered, so widening cannot rescue heavy-tail coverage (max achievable ~2.8 e-folds at the exp-era candidate). The beta measurement is conditional on the truncated window; A-vs-B agreement cannot detect window-induced bias because both routes see the same window.

## beta-table row candidate

| burst | beta | alpha (derived) | tau_1GHz [ms] | grade |
|---|---|---|---|---|
| freya | 3.684 +0.013/-0.014 | 4.376 +0.019/-0.018 | 0.1144 +/- 0.0011 | PASS |

Provenance: `analysis/scattering-refit-2026-06/local_runs/freya_joint_fit_sharedzeta.json` and siblings; DAG #99 #100 #101 #102 #103 #104(e09ac78b) #105(b8c8ffe5) -> #106.
