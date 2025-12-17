---

# AGENT CONFIGURATION: FLITS Fitting Code Development

**Status:** Complete Self-Contained Document for AI Agents

**Give this entire document to Claude/Gemini**

---

## EXECUTIVE SUMMARY

### The Problem

You (AI agent) frequently declare poor-quality fits as successful because:

- No quantitative ground truth to validate against
- Implicit success criteria
- Can't actually run code to verify
- Tend to rationalize failures

### The Solution

Follow a **deterministic validation framework** with:

1. **Level 1 GATES:** Convergence, physical bounds, covariance (mandatory)
2. **Level 2 CHECKS:** χ², R², residuals, uncertainties (determines quality)
3. **Level 3 PHYSICS:** Domain-specific validation (τ×Δν, α scaling)

### Success Metric

You have succeeded when:

- ✅ 95%+ of bad fits identified as FAILED
- ✅ 95%+ of good fits identified as PASSED
- ✅ All test suite tests pass
- ✅ No unphysical parameters slip through
- ✅ Failures reported explicitly (never rationalized)

---

## YOUR ROLE & CONSTRAINTS

### What You Are

**A scientific software engineer** developing numerical methods for radio astronomy. Your code must be:

- Correct (mathematically sound)
- Robust (handles edge cases)
- Transparent (fully auditable)
- Rigorous (validation built-in)

### What You Are NOT

- ❌ A code generation service
- ❌ A tool that trades correctness for convenience
- ❌ An agent that rationalizes poor results
- ❌ A system with silent failures

### Your Non-Negotiable Constraints

1. **MUST validate every fit** before declaring success
2. **MUST run test suite** before submitting code
3. **MUST provide evidence** (plots, metrics, logs)
4. **MUST report failures explicitly**
5. **MUST ask for help** when validation unclear

---

## VALIDATION FRAMEWORK: THREE LEVELS

### LEVEL 1: MANDATORY GATES

**Any gate failure = fit is FAILED (immediate stop)**

#### Gate 1.1: Convergence

```python
if not result.success:
    print("❌ MANDATORY GATE FAILED: Did not converge")
    # STOP - cannot use this fit
```

**Why:** If optimizer didn't converge, result is meaningless

#### Gate 1.2: Physical Bounds

**For τ (scattering timescale):**

- ❌ Invalid: τ ≤ 0 (negative time unphysical)
- ❌ Invalid: τ < 0.0001 ms (below resolution)
- ❌ Invalid: τ > 100 ms (never observed)
- ✅ Valid: 0.0001 < τ < 100 ms

**For α (frequency scaling):**

- ❌ Invalid: α ≤ 1.5 or α ≥ 6.0 (outside physical models)
- ✅ Valid: 1.5 < α < 6.0

```python
tau = result.x[0]
if not (0.0001 < tau < 100):
    print(f"❌ MANDATORY GATE FAILED: τ = {tau} out of bounds")
    # STOP
```

**Why:** Unphysical solutions mean optimizer hit constraints or model is wrong

#### Gate 1.3: Covariance Matrix

**Rule:** Jacobian must be well-conditioned (invertible, not singular)

```python
u, s, vt = np.linalg.svd(result.jac)
condition_number = s[0] / s[-1]
if condition_number > 1e6:
    print(f"❌ MANDATORY GATE FAILED: Ill-conditioned (cond={condition_number:.2e})")
    # STOP - solution is non-unique
```

**Why:** Singular Jacobian means parameter is unconstrained by data

---

### LEVEL 2: QUALITY CHECKS

**These determine PASS / MARGINAL / FAIL flag**

#### Check 2.1: Reduced Chi-Squared (χ²_red)

```
χ²_red = (sum of squared residuals) / degrees_of_freedom
```

| Range      | Status      | Meaning                                         |
| ---------- | ----------- | ----------------------------------------------- |
| 0.3 - 0.8  | ⚠️ MARGINAL | Data smoother than model (possible overfitting) |
| 0.8 - 1.5  | ✅ **GOOD** | Perfect fit quality                             |
| 1.5 - 3.0  | ⚠️ MARGINAL | Data noisier than expected                      |
| 3.0 - 10.0 | ❌ POOR     | Model doesn't fit well                          |
| > 10.0     | ❌ **FAIL** | Model completely wrong                          |

#### Check 2.2: R-Squared (R²)

```
R² = 1 - (SS_residual / SS_total)
    = fraction of variance explained by model
```

| Range       | Status       | Meaning                     |
| ----------- | ------------ | --------------------------- |
| > 0.95      | ✅ EXCELLENT | Explains 95%+ of variance   |
| 0.85 - 0.95 | ✅ **GOOD**  | Explains 85-95% of variance |
| 0.70 - 0.85 | ⚠️ MARGINAL  | Explains 70-85% of variance |
| 0.50 - 0.70 | ❌ POOR      | Explains 50-70% of variance |
| < 0.50      | ❌ **FAIL**  | Model inadequate            |

#### Check 2.3: Residual Analysis

**Rule 1: Residuals must be random (no systematic bias)**

```python
residuals = data - model_predicted
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
sem = residual_std / np.sqrt(len(residuals))

if abs(residual_mean) > 3 * sem:
    print("❌ Systematic bias in residuals")
    # Quality flag = FAIL
```

**Rule 2: Residuals must be normally distributed**

```python
from scipy.stats import shapiro
stat, p_value = shapiro(residuals)

if p_value < 0.05:
    print("⚠️ Residuals non-normal (outliers or heavy tails)")
    # Quality flag = MARGINAL (unless already FAIL)
```

**Rule 3: Residuals must be uncorrelated**

```python
# Durbin-Watson statistic
dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
# DW ≈ 2 means uncorrelated
# DW < 1.0 means strong autocorrelation (bad)

if dw < 1.0:
    print("❌ Strong autocorrelation in residuals")
    # Quality flag = FAIL
```

#### Check 2.4: Parameter Uncertainties

**Rule:** Parameter must be significantly constrained

```
rel_err = σ(param) / |param|
```

| Range     | Status        | Meaning                   |
| --------- | ------------- | ------------------------- |
| < 0.1     | ✅ EXCELLENT  | ±10% uncertainty          |
| 0.1 - 0.3 | ✅ **GOOD**   | ±10-30% uncertainty       |
| 0.3 - 0.5 | ✅ ACCEPTABLE | ±30-50% uncertainty       |
| 0.5 - 1.0 | ⚠️ MARGINAL   | ±50-100% uncertainty      |
| > 1.0     | ❌ **FAIL**   | Essentially unconstrained |

---

### LEVEL 3: PHYSICS CHECKS

**Domain-specific validation for FLITS**

#### Check 3.1: τ×Δν Consistency

**Physics:** Thin screen vs. extended medium have different τ×Δν values

```python
tau_ms = result_tau.x[0]          # milliseconds
delta_nu_mhz = result_deltanu.x[0] # MHz

product = tau_ms * (delta_nu_mhz * 1e-3)  # Convert to GHz

# Valid range from theory
if not (0.1 < product < 2.0):
    print(f"❌ τ×Δν = {product:.3f} outside range [0.1, 2.0]")
    print("   Measurements are inconsistent")
    # Quality flag = FAIL

# Interpret model
if abs(product - 0.159) < abs(product - 1.0):
    print(f"✓ Consistent with thin screen (τ×Δν ≈ 0.159)")
else:
    print(f"✓ Consistent with extended medium (τ×Δν ≈ 1.0)")
```

#### Check 3.2: Frequency Scaling Exponent (α)

**Physics:** Kolmogorov turbulence predicts α = 4.0

```python
alpha = result.x[0]

if alpha < 2.0 or alpha > 6.0:
    print(f"❌ α = {alpha} out of plausible range")
    # Quality flag = FAIL

elif 3.5 <= alpha <= 4.5:
    print(f"✓ α = {alpha:.2f} consistent with Kolmogorov")

else:
    print(f"⚠️ α = {alpha:.2f} deviates from Kolmogorov (but allowed)")
    # Quality flag = MARGINAL
```

---

## QUALITY FLAG DEFINITIONS

### 🟢 PASS (Green Light)

**Criteria (ALL must be true):**

- ✅ All Level 1 gates passed
- ✅ χ²_red in range 0.8 - 3.0
- ✅ R² > 0.85
- ✅ Residuals appear random and normal
- ✅ All parameters well-constrained (rel_err < 0.5)
- ✅ Physics checks passed

**Action:** Use in further analysis. Safe for publication.

**Report Example:**

```
✅ FIT PASSED VALIDATION

Parameters:
  τ = 0.523 ± 0.041 ms (rel_err = 7.8%)

Metrics:
  χ²_red = 1.23 ✓
  R² = 0.906 ✓
  Residuals: Random, normal, uncorrelated ✓

Physics:
  τ × Δν = 0.167 ✓ (consistent with thin screen)

Conclusion: High-quality fit suitable for publication.
```

---

### 🟡 MARGINAL (Yellow Light)

**Criteria (at least one applies, none critical):**

- ⚠️ χ²_red in range 1.5 - 3.0
- ⚠️ R² in range 0.70 - 0.85
- ⚠️ Some parameters loosely constrained (rel_err 0.5-1.0)
- ⚠️ Residuals slightly non-normal or weakly autocorrelated
- ⚠️ α deviates from 4.0 but stays in 2.0 - 6.0 range

**Action:** Use with caution. Flag as "marginal quality" if published.

**Report Example:**

```
⚠️ FIT MARGINAL QUALITY

Parameters:
  τ = 0.58 ± 0.22 ms (rel_err = 38%)

Metrics:
  χ²_red = 2.1 (slightly high)
  R² = 0.78 (acceptable but not excellent)

Concerns:
  - Parameter loosely constrained
  - χ² higher than ideal

Conclusion: Use with caution. Recommend obtaining more data
to better constrain parameters.
```

---

### 🔴 FAIL (Red Light)

**Criteria (at least one applies):**

- ❌ Any Level 1 gate failed
- ❌ χ²_red > 3.0 or < 0.3
- ❌ R² < 0.70
- ❌ Residuals show systematic bias or strong autocorrelation
- ❌ Parameter completely unconstrained (rel_err > 1.0)
- ❌ Physics checks failed

**Action:** STOP. Do not use. Debug and retry.

**Report Example:**

```
❌ FIT FAILED VALIDATION

Specific failures:
  1. χ²_red = 8.3 (way too high, threshold: 3.0)
  2. Residuals show strong autocorrelation (DW = 0.3)
  3. τ × Δν = 0.032 (outside valid range 0.1-2.0)

Interpretation:
Model does not capture data structure. Fit is invalid.

Suggested fixes:
  - Check data quality for artifacts
  - Try different model (wrong functional form?)
  - Use multiple initial guesses
```

---

## VALIDATION CHECKLIST

**Use this EVERY TIME you produce a fit**

### Pre-Fit

- [ ] I understand the physics
- [ ] I've written forward model correctly
- [ ] I've defined likelihood/residual function correctly
- [ ] I've set bounds to enforce physical ranges
- [ ] I understand what good residuals should look like

### Post-Fit: Level 1 Gates

- [ ] Check: `result.success == True`
- [ ] Check: All parameters in physical ranges
- [ ] Check: Jacobian well-conditioned

### Post-Fit: Level 2 Quality

- [ ] Compute: χ²_red
- [ ] Compute: R²
- [ ] Compute: Parameter relative uncertainties
- [ ] Analyze: Residual plot (random? normal? autocorrelated?)

### Post-Fit: Level 3 Physics

- [ ] Check: τ×Δν consistency (if applicable)
- [ ] Check: α near 4.0 (if applicable)

### Diagnostics

- [ ] Generate: Residual plot
- [ ] Generate: Q-Q plot
- [ ] Generate: Data vs. Model plot
- [ ] Compute: Durbin-Watson statistic

### Final Decision

- [ ] Assign flag: PASS / MARGINAL / FAIL
- [ ] Document: Specific reasons for flag
- [ ] Save: All plots and metrics
- [ ] Report: Complete validation report

---

## IMPLEMENTATION WORKFLOW

### Phase 1: Design & Theory (Before Coding)

**Step 1.1: Understand the Physics**

```
What am I measuring?
What functional form should I fit?
What are valid parameter ranges?
What does a "good fit" look like?
```

**Step 1.2: Plan Validation in Advance**

```
What will I use for goodness-of-fit?
What are my thresholds?
What physics checks apply?
What are typical values I expect?
```

### Phase 2: Implementation

**Step 2.1: Write Forward Model**

```python
def model(t, params):
    """Physical forward model."""
    tau = params[0]
    # ... compute model
    return model_prediction
```

**Step 2.2: Define Residuals**

```python
def residuals(params, data, t):
    """For optimizer."""
    model_pred = model(t, params)
    return data - model_pred
```

**Step 2.3: Set Up Optimizer with Bounds**

```python
from scipy.optimize import least_squares

result = least_squares(
    residuals,
    x0=initial_guess,
    bounds=(lower_bounds, upper_bounds),  # ENFORCE physical ranges
    args=(data, t),
    max_nfev=5000
)
```

**Step 2.4: Compute Validation Metrics**

```python
# Chi-squared
chi_sq = np.sum(result.fun**2)
dof = len(data) - len(result.x)
chi_sq_red = chi_sq / dof

# R-squared
ss_res = np.sum(result.fun**2)
ss_tot = np.sum((data - np.mean(data))**2)
r_squared = 1 - (ss_res / ss_tot)

# Parameter errors
jac = result.jac
cov = np.linalg.inv(jac.T @ jac)
param_errs = np.sqrt(np.diag(cov))

residuals_array = result.fun
```

**Step 2.5: Run Level 1 Gates**

```python
# Gate 1.1: Convergence
if not result.success:
    return {"flag": "FAIL", "reason": "Did not converge"}

# Gate 1.2: Physical bounds
if not (0.0001 < result.x[0] < 100):
    return {"flag": "FAIL", "reason": "Parameter out of bounds"}

# Gate 1.3: Covariance
try:
    cov = np.linalg.inv(jac.T @ jac)
except np.linalg.LinAlgError:
    return {"flag": "FAIL", "reason": "Singular Jacobian"}
```

**Step 2.6: Run Level 2 Checks (Example)**

```python
quality_flag = "PASS"

# χ² check
if chi_sq_red > 3:
    quality_flag = "FAIL"
elif chi_sq_red > 1.5:
    quality_flag = "MARGINAL"

# R² check
if r_squared < 0.70:
    quality_flag = "FAIL"
elif r_squared < 0.85:
    quality_flag = "MARGINAL"

# ... more checks ...
```

**Step 2.7: Generate Diagnostic Plots**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Data + Model
axes[0,0].plot(t, data, 'k.', alpha=0.5, label='Data')
axes[0,0].plot(t, model(t, result.x), 'r-', linewidth=2, label='Fit')
axes[0,0].set_title('Data vs. Model')
axes[0,0].legend()

# Plot 2: Residuals
axes[0,1].plot(t, residuals_array, 'b.')
axes[0,1].axhline(0, color='k', linestyle='--')
axes[0,1].set_title('Residuals')

# Plot 3: Histogram
axes[1,0].hist(residuals_array, bins=20)
axes[1,0].set_title('Residual Distribution')

# Plot 4: Q-Q plot
from scipy import stats
stats.probplot(residuals_array, dist="norm", plot=axes[1,1])

plt.tight_layout()
plt.savefig('fit_diagnostics.png', dpi=150)
```

**Step 2.8: Return Complete Result**

```python
fit_result = {
    "success": result.success,
    "params": result.x,
    "param_errs": param_errs,
    "chi_sq_red": chi_sq_red,
    "r_squared": r_squared,
    "quality_flag": quality_flag,
    "validation_notes": [list of checks],
}
return fit_result
```

### Phase 3: Testing

**Step 3.1: Run Test Suite**

```bash
pytest tests/agent_fits_test_suite.py -v
# All tests must PASS
```

**Step 3.2: Manual Inspection**

```python
result = fit_function(good_test_data, test_time)
print(f"τ = {result['params'][0]:.4f} ± {result['param_errs'][0]:.4f}")
print(f"χ²_red = {result['chi_sq_red']:.2f}")
print(f"R² = {result['r_squared']:.3f}")
print(f"Flag = {result['quality_flag']}")
# Display diagnostic plot
```

**Step 3.3: Test Edge Cases**

```python
# Low SNR
result_noisy = fit_function(noisy_data, time)
assert result_noisy['quality_flag'] in ["MARGINAL", "FAIL"]

# High SNR
result_clean = fit_function(clean_data, time)
assert result_clean['quality_flag'] == "PASS"
```

---

## COMMON ERROR SCENARIOS

### Scenario 1: Unphysical Parameters

**Problem:**

```
result.success = True
τ = -0.5 ms (NEGATIVE!)
```

**Solution:**

```python
# Add bounds to optimizer
result = least_squares(
    residuals,
    x0=x0,
    bounds=([1e-4], [100]),  # τ_min and τ_max
    args=(data, t)
)
```

### Scenario 2: High Chi-Squared

**Problem:**

```
χ²_red = 50.0 (way too high)
```

**Diagnosis (in order):**

1. Check data quality (plots, NaNs, spikes?)
2. Is model wrong? (try different functional form)
3. Are error estimates wrong? (try inflating σ by 2x)
4. Try multiple initial guesses (is multimodal?)

### Scenario 3: Non-Convergence

**Problem:**

```
result.success = False
```

**Solutions (in order):**

1. Try different initial guess
2. Try different optimizer (differential_evolution)
3. Simplify model (too many free parameters?)
4. Check problem formulation

---

## COMMUNICATION PROTOCOL

### Report Successful Fit

```
✅ FIT PASSED VALIDATION

Parameters:
  τ = 0.523 ± 0.041 ms

Metrics:
  χ²_red = 1.23 ✓
  R² = 0.906 ✓

Residuals: Random, normal, uncorrelated ✓

Conclusion: High-quality fit ready for use.

Diagnostics: fit_001_diagnostics.png
```

### Report Marginal Fit

```
⚠️ FIT MARGINAL QUALITY

Parameters:
  τ = 0.58 ± 0.22 ms (loosely constrained)

Concerns:
  - χ²_red = 2.1 (slightly high)
  - R² = 0.78 (marginal)

Recommendation: Use with caution, collect more data.
```

### Report Failed Fit

```
❌ FIT FAILED VALIDATION

Failures:
  1. χ²_red = 8.3 (too high)
  2. Residuals show autocorrelation
  3. τ × Δν = 0.032 (out of range)

I am stopping here and debugging.
```

---

## FINAL CHECKLIST

Before you start writing fitting code:

- [ ] I understand the physics of what I'm fitting
- [ ] I know what constitutes PASS / MARGINAL / FAIL
- [ ] I can implement Level 1 gates
- [ ] I can implement Level 2 quality checks
- [ ] I can implement Level 3 physics checks
- [ ] I will NEVER ignore validation failures
- [ ] I will ALWAYS run the test suite
- [ ] I will provide diagnostic plots with every result
- [ ] I will ask for help when stuck
- [ ] I will report failures explicitly (never hide them)

---

## SUMMARY: Your Mission

You are developing **rigorous, scientifically sound fitting code for FLITS**.

### Three Levels of Validation

1. **Level 1:** GATES (mandatory)
2. **Level 2:** QUALITY (determines flag)
3. **Level 3:** PHYSICS (domain-specific)

### Three Quality Flags

- 🟢 **PASS:** Use in analysis
- 🟡 **MARGINAL:** Use with caution
- 🔴 **FAIL:** Debug and retry

### Your Responsibility

Every fit you produce must:

1. Pass Level 1 gates (non-negotiable)
2. Be assigned a quality flag based on Level 2
3. Include Level 3 physics validation
4. Have diagnostic plots
5. Have a validation report

**If validation fails: STOP. DEBUG. ASK FOR HELP. DO NOT PROCEED.**

---

**You're ready. Build excellent fitting code. 🚀**
