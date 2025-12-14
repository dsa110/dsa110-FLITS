# FLITS Environment Setup Instructions

Due to SSL certificate issues with the current conda installation, here are alternative methods to create the `flits` environment:

## Method 1: Manual Installation (Recommended)

```bash
# Activate your existing working Python 3.12 environment
conda activate dsa_contimg  # or your current environment

# Install FLITS dependencies with pip
pip install numpy scipy matplotlib pandas astropy h5py pyyaml tqdm emcee corner lmfit ipython jupyter notebook ipywidgets

# Install FLITS package in development mode
cd /data/jfaber/FLITS
pip install -e .
```

## Method 2: Fix Conda SSL Issues

If you want to fix conda and create a dedicated environment:

```bash
# Reinstall/upgrade urllib3 to fix SSL issues
pip install --upgrade urllib3

# Remove conda-forge from channels temporarily
conda config --remove channels conda-forge

# Try creating environment again
conda create -n flits python=3.12 -y

# Once created, activate and install packages
conda activate flits
conda install -c defaults numpy scipy matplotlib pandas jupyter ipython notebook ipywidgets -y
pip install astropy h5py pyyaml tqdm emcee corner lmfit
```

## Method 3: Use Existing Environment

Since you already have `dsa_contimg` with Python 3.12, you can just install the missing packages:

```bash
conda activate dsa_contimg
pip install emcee corner lmfit  # These are likely the only missing packages
```

## Required Packages

### Core Scientific:
- numpy >= 1.24
- scipy >= 1.11
- matplotlib >= 3.7
- pandas >= 2.0

### Astronomy:
- astropy >= 5.3

### Data I/O:
- h5py >= 3.9
- pyyaml >= 6.0

### MCMC & Fitting:
- emcee >= 3.1
- corner >= 2.2
- lmfit >= 1.2

### Jupyter:
- ipython >= 8.14
- jupyter >= 1.0
- notebook >= 7.0
- ipywidgets >= 8.1

### Utilities:
- tqdm >= 4.65

## Verify Installation

```python
# Test that everything works
import numpy
import scipy
import matplotlib
import emcee
import corner
import lmfit
import ipywidgets

print("✓ All packages imported successfully!")
```

## Note

The `environment.yml` file has been created at `/data/jfaber/FLITS/environment.yml` for when conda SSL issues are resolved.
