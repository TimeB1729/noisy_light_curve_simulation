# Noisy Light Curve Simulation using Gaussian Processes

## Overview

This project demonstrates how Gaussian Processes (GPs) can be used to model, simulate, and impute noise in stellar light curves, especially in the context of astrophysical observations such as those from the Kepler Mission. The notebook includes both theoretical underpinnings and practical implementations of Gaussian Process Regression (GPR), and it builds up to working with real, incomplete astronomical data.

## 🧭 Notebook Sections

### 1. 📘 Gaussian Process Regression — Theory & Experimentation

Introduces the concept of GPR, emphasizing the squared exponential (RBF) kernel, and explores how GPs define distributions over functions. This section includes:

- Manual construction of RBF kernels.
- Sampling functions from the GP prior.
- Visual demonstrations of the effect of hyperparameters (A, l).

### 2. 🔬 Simulating a Noisy Light Curve

This section builds a synthetic light curve by:

- Generating a true signal (sinusoidal + Gaussian pulse).
- Adding GP-simulated correlated noise.
- Injecting white noise.

The result is a synthetic but realistic stellar light curve—ideal for testing denoising or inference algorithms.

### 3. 🔭 Real Kepler Light Curves and GP-Based Inference

Two light curves from the Kepler Mission are analyzed:

- KIC 007596240
(A relatively quiet star with 71,427 flux measurements, ~16% missing. GP is used to impute the missing values while preserving the statistical structure of the light curve.)

- KIC 007609553
(A more active, rotating star showing periodic brightness modulations due to starspots. GP is used to model both the variability and the uncertainties.)

## 📈 Applications

- Imputation of missing astronomical data

- Simulation of realistic observational noise

- Testing preprocessing pipelines before applying inference

- Understanding the role of correlated noise in astrophysical signals

## 📂 Dataset Notes

- ['Kepler1.dat'](notebook/datasets/Kepler1.dat):
Quiet star (KIC 007596240) with 16% missing data. Used for interpolation using GP.

- ['Kepler2.dat'](notebook/datasets/Kepler2.dat):
Active rotating star (KIC 007609553). GP captures periodic variability and noise.

## 🤝 Acknowledgements

This is an **independent project** developed by **Shyam Banerjee** (B.Stat., ISI Kolkata) as part of a summer exploration in astrostatistical modeling. The core ideas and motivation were inspired by the lecture notes from the **Summer School for Astrostatistics** at Penn State University, particularly the sessions by **Dr. Suzanne Aigrain** on Gaussian Processes in Astronomy.

The datasets are downloaded from the `NASA/IPAC/NExScI Star And Exoplanet` archives.

## ⚙️ Running the Notebooks

All notebooks are written in Python and run on `Jupyter`. You can install the required packages with:

```bash
pip install -r requirements.txt