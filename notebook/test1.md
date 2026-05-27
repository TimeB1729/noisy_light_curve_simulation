# Testing file 1

## Stellar light curves from the Keplar Mission



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data (handle NA safely) and save memory using float32
flux1 = np.genfromtxt("datasets/Kepler1.dat", dtype=np.float32, missing_values="NA", filling_values=np.nan)
flux2 = np.genfromtxt("datasets/Kepler2.dat", dtype=np.float32, missing_values="NA", filling_values=np.nan)

# Count missing values
n_missing_flux1 = np.isnan(flux1).sum()
n_missing_flux2 = np.isnan(flux2).sum()
n_total_flux1 = len(flux1)
n_total_flux2 = len(flux2)
print(f"Kepler1: {n_missing_flux1}/{n_total_flux1} ({100 * n_missing_flux1 / n_total_flux1:.2f}% missing)")
print(f"kepler2: {n_missing_flux2}/{n_total_flux2} ({100 * n_missing_flux2 / n_total_flux2:.2f}% missing)")
```

    Kepler1: 11337/71427 (15.87% missing)
    kepler2: 20598/71427 (28.84% missing)



```python
# Masked plots to ignore NaNs
t1 = np.arange(len(flux1))
t2 = np.arange(len(flux2))

plt.figure(figsize=(12, 4))
plt.plot(t1[~np.isnan(flux1)], flux1[~np.isnan(flux1)], lw=0.4, label='Kepler1: Quiet Star')
plt.plot(t2[~np.isnan(flux2)], flux2[~np.isnan(flux2)], lw=0.8, label='Kepler2: Rotating Star')
plt.xlabel("Time Index")
plt.ylabel("Flux")
plt.legend()
plt.title("Kepler Light Curves")
plt.grid(True)
plt.show()
```


    
![png](test1_files/test1_3_0.png)
    



```python
from scipy.interpolate import interp1d

# Only use valid indices
valid1 = ~np.isnan(flux1)
interp_func1 = interp1d(t1[valid1], flux1[valid1], kind='linear', fill_value="extrapolate")
flux1_interp = interp_func1(t1)

valid2 = ~np.isnan(flux2)
interp_func2 = interp1d(t2[valid2], flux2[valid2], kind = 'linear', fill_value = 'extrapolate')
flux2_interp = interp_func2(t2)
```


```python
from scipy.optimize import curve_fit

def sinusoid(t, A, P, phi, offset):
    return A * np.sin(2 * np.pi * t / P + phi) + offset

# Use only finite values
finite = ~np.isnan(flux2)
t_fit = t2[finite]
f_fit = flux2[finite] - np.nanmedian(flux2)

# Initial guesses: A=10, P=50, phi=0, offset=0
popt, _ = curve_fit(sinusoid, t_fit, f_fit, p0=[10, 50, 0, 0])
A, P, phi, offset = popt

# Reconstruct fitted signal
fitted = sinusoid(t_fit, *popt)

plt.figure(figsize=(10, 4))
plt.plot(t_fit, f_fit, label='Flux2', alpha=0.5)
plt.plot(t_fit, fitted, label='Fitted Sinusoid', color='r')
plt.legend()
plt.title(f"Kepler2 Fit: A={A:.2f}, P={P:.2f}")
plt.grid(True)
plt.show()
```


    
![png](test1_files/test1_5_0.png)
    


## Using Gaussian Processes to simulate probable noise


```python
np.random.seed(442)

def man_ExpSquaredKernel(t1, t2=None, A=1.0, l=1.0):
    if t2 is None:
        t2 = t1
    T2, T1 = np.meshgrid(t2, t1)
    return A ** 2 * np.exp(-0.5 * (T1-T2)**2 / l**2)

N = 1000
t1_sub = t1[:N]
flux1_sub = flux1_interp[:N]

cov1_sub = man_ExpSquaredKernel(t1_sub, A = 0.5, l = 50.0)
gp_noise_sub1 = np.random.multivariate_normal(mean=np.zeros(N), cov=cov1_sub)

# Add GP noise
flux_gp_injected1 = flux1_sub + gp_noise_sub1

plt.figure(figsize=(12, 5))
plt.plot(t1_sub, flux_gp_injected1, label='Injected GP Noise', lw = 0.4)
plt.plot(t1_sub, flux1_sub, '--', label='Original Flux', alpha=0.5)
plt.xlabel("Time Index")
plt.ylabel("Flux")
plt.legend()
plt.title("Simulated GP Noise on Kepler1 (First N = 1000 points)")
plt.grid(True)
plt.show()
```


    
![png](test1_files/test1_7_0.png)
    



```python
np.random.seed(442)

N = 1000
t2_sub = t2[:N]
flux2_sub = flux2_interp[:N]

cov2_sub = man_ExpSquaredKernel(t2_sub, A = 0.5, l = 50.0)
gp_noise_sub2 = np.random.multivariate_normal(mean=np.zeros(N), cov=cov2_sub)

# Add GP noise
flux_gp_injected2 = flux2_sub + gp_noise_sub2

plt.figure(figsize=(12, 5))
plt.plot(t2_sub, flux_gp_injected2, label='Injected GP Noise', lw = 0.4)
plt.plot(t2_sub, flux2_sub, '--', label='Original Flux', alpha=0.5)
plt.xlabel("Time Index")
plt.ylabel("Flux")
plt.legend()
plt.title("Simulated GP Noise on Kepler2 (First N = 1000 points)")
plt.grid(True)
plt.show()
```


    
![png](test1_files/test1_8_0.png)
    


## Modeling around Kepler datas using GPR

### Kepler 1 imputation


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import interp1d
import IPython.display as display

# Step 1: Load flux data (Note: ensure path matches your environment)
flux1 = np.genfromtxt("datasets/Kepler1.dat", dtype=np.float32, missing_values="NA", filling_values=np.nan)

# Step 2: Create time index
t1 = np.arange(len(flux1))

# Step 3: Get observed (non-NaN) values ONLY
mask = ~np.isnan(flux1)
t1_valid = t1[mask]
y1_valid = flux1[mask]

# Step 3.5: Subsample the flux data before split to manage O(N^3) memory constraints
subset = 100
t1_sub = t1_valid[::subset]
y1_sub = y1_valid[::subset]

# Step 3.6: Randomized Train-Test Split (80% Train, 20% Test)

# --- 1. Setup Monte Carlo Cross-Validation ---
n_repetitions = 20
seeds = range(n_repetitions)

# Storage for GP metrics
gp_rmse_list, gp_mae_list, gp_r2_list = [], [], []
gp_1sigma_list, gp_2sigma_list, gp_std_list = [], [], []

# Storage for Baseline metrics
lin_rmse_list, lin_mae_list, lin_r2_list = [], [], []
cub_rmse_list, cub_mae_list, cub_r2_list = [], [], []

print(f"Running Monte Carlo Cross-Validation across {n_repetitions} seeds...")

kernel1 = 1.0 * RBF(length_scale=100.0) + WhiteKernel(noise_level=1.0)

for seed in seeds:
    # Randomized Split
    t1_train, t1_test, y1_train, y1_test = train_test_split(
        t1_sub, y1_sub, test_size=0.2, random_state=seed
    )

    # Sort arrays for clean interpolation
    sort_train = np.argsort(t1_train)
    t1_train, y1_train = t1_train[sort_train], y1_train[sort_train]
    sort_test = np.argsort(t1_test)
    t1_test, y1_test = t1_test[sort_test], y1_test[sort_test]

    X1_train, X1_test = t1_train.reshape(-1, 1), t1_test.reshape(-1, 1)

    # --- Fit & Evaluate GP ---
    gpr1 = GaussianProcessRegressor(kernel=kernel1, alpha=1e-2, normalize_y=True, random_state=seed)
    gpr1.fit(X1_train, y1_train)
    y1_pred, y1_std = gpr1.predict(X1_test, return_std=True)

    gp_rmse_list.append(np.sqrt(mean_squared_error(y1_test, y1_pred)))
    gp_mae_list.append(mean_absolute_error(y1_test, y1_pred))
    gp_r2_list.append(r2_score(y1_test, y1_pred))
    gp_std_list.append(np.mean(y1_std))

    # Uncertainty Coverage
    gp_1sigma_list.append(np.mean((y1_test >= y1_pred - y1_std) & (y1_test <= y1_pred + y1_std)))
    gp_2sigma_list.append(np.mean((y1_test >= y1_pred - 2*y1_std) & (y1_test <= y1_pred + 2*y1_std)))

    # --- Fit & Evaluate Baselines ---
    # Linear
    lin_interp = interp1d(t1_train, y1_train, kind='linear', fill_value='extrapolate')
    y_pred_lin = lin_interp(t1_test)
    lin_rmse_list.append(np.sqrt(mean_squared_error(y1_test, y_pred_lin)))
    lin_mae_list.append(mean_absolute_error(y1_test, y_pred_lin))
    lin_r2_list.append(r2_score(y1_test, y_pred_lin))

    # Cubic
    cub_interp = interp1d(t1_train, y1_train, kind='cubic', fill_value='extrapolate')
    y_pred_cub = cub_interp(t1_test)
    cub_rmse_list.append(np.sqrt(mean_squared_error(y1_test, y_pred_cub)))
    cub_mae_list.append(mean_absolute_error(y1_test, y_pred_cub))
    cub_r2_list.append(r2_score(y1_test, y_pred_cub))

print("Monte Carlo evaluation complete.")
```

    Running Monte Carlo Cross-Validation across 20 seeds...
    Monte Carlo evaluation complete.



```python
# --- 1. Compute Aggregates ---
benchmark_data = {
    "Method": ["Linear Interpolation", "Cubic Spline", "Gaussian Process"],
    "RMSE Mean": [np.mean(lin_rmse_list), np.mean(cub_rmse_list), np.mean(gp_rmse_list)],
    "RMSE Std": [np.std(lin_rmse_list), np.std(cub_rmse_list), np.std(gp_rmse_list)],
    "MAE Mean": [np.mean(lin_mae_list), np.mean(cub_mae_list), np.mean(gp_mae_list)],
    "MAE Std": [np.std(lin_mae_list), np.std(cub_mae_list), np.std(gp_mae_list)],
    "R² Mean": [np.mean(lin_r2_list), np.mean(cub_r2_list), np.mean(gp_r2_list)],
    "R² Std": [np.std(lin_r2_list), np.std(cub_r2_list), np.std(gp_r2_list)]
}

benchmark_df = pd.DataFrame(benchmark_data)

print("--- Kepler 1 Benchmark Results (20 Seeds) ---")
display.display(benchmark_df.round(4))

# --- 2. Uncertainty Calibration Summary ---
print("\n--- GP Uncertainty Calibration Summary ---")
print(f"Average 1σ Coverage: {np.mean(gp_1sigma_list)*100:.2f}% ± {np.std(gp_1sigma_list)*100:.2f}% (Ideal: ~68%)")
print(f"Average 2σ Coverage: {np.mean(gp_2sigma_list)*100:.2f}% ± {np.std(gp_2sigma_list)*100:.2f}% (Ideal: ~95%)")

# --- 3. Best/Worst Case Analysis ---
best_seed_idx = np.argmin(gp_rmse_list)
worst_seed_idx = np.argmax(gp_rmse_list)
print("\n--- Sensitivity to Train-Test Split ---")
print(f"Best GP RMSE:  {gp_rmse_list[best_seed_idx]:.4f} (Seed {seeds[best_seed_idx]})")
print(f"Worst GP RMSE: {gp_rmse_list[worst_seed_idx]:.4f} (Seed {seeds[worst_seed_idx]})")
```

    --- Kepler 1 Benchmark Results (20 Seeds) ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>RMSE Mean</th>
      <th>RMSE Std</th>
      <th>MAE Mean</th>
      <th>MAE Std</th>
      <th>R² Mean</th>
      <th>R² Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Interpolation</td>
      <td>6.2827</td>
      <td>0.3437</td>
      <td>5.1008</td>
      <td>0.3127</td>
      <td>-0.5733</td>
      <td>0.1151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cubic Spline</td>
      <td>8.6941</td>
      <td>1.1385</td>
      <td>6.6883</td>
      <td>0.5017</td>
      <td>-2.0335</td>
      <td>0.7128</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gaussian Process</td>
      <td>5.0477</td>
      <td>0.2525</td>
      <td>4.1156</td>
      <td>0.2176</td>
      <td>-0.0136</td>
      <td>0.0182</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- GP Uncertainty Calibration Summary ---
    Average 1σ Coverage: 65.12% ± 4.24% (Ideal: ~68%)
    Average 2σ Coverage: 96.07% ± 1.63% (Ideal: ~95%)
    
    --- Sensitivity to Train-Test Split ---
    Best GP RMSE:  4.6718 (Seed 11)
    Worst GP RMSE: 5.5869 (Seed 7)



```python
# --- Plot RMSE Distribution ---
plt.figure(figsize=(8, 5))
plt.hist(gp_rmse_list, bins=10, color='royalblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(gp_rmse_list), color='red', linestyle='dashed', linewidth=2, label=f'Mean RMSE: {np.mean(gp_rmse_list):.4f}')
plt.title(f"Distribution of GP RMSE across {n_repetitions} Random Splits", fontsize=14)
plt.xlabel("Root Mean Squared Error (RMSE)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```


    
![png](test1_files/test1_13_0.png)
    



```python
# --- Visualization of the Best GP Model Split for Kepler 1 ---

# Step 1: Identify the best performing seed for Kepler 1
best_k1_idx = np.argmin(gp_rmse_list)
best_k1_seed = seeds[best_k1_idx]

print(f"Generating visualization for Best Kepler 1 Seed: {best_k1_seed} (RMSE: {gp_rmse_list[best_k1_idx]:.4f})")

# Step 2: Recreate the train/test split for that specific best seed
t1_train_best, t1_test_best, y1_train_best, y1_test_best = train_test_split(
    t1_sub, y1_sub, test_size=0.2, random_state=best_k1_seed
)

X1_train_best = t1_train_best.reshape(-1, 1)

# Step 3: Retrain the GP model on the best split
gpr1_best = GaussianProcessRegressor(kernel=kernel1, alpha=1e-2, normalize_y=True, random_state=best_k1_seed)
gpr1_best.fit(X1_train_best, y1_train_best)

# Step 4: Predict over the FULL original time range to generate the smooth background band
X1_all = t1.reshape(-1, 1)
y1_pred_all_best, y1_std_all_best = gpr1_best.predict(X1_all, return_std=True)

# Step 5: Publication-style Plotting
plt.figure(figsize=(14, 6))

# Plot full predictive band for the winning model
plt.plot(t1, y1_pred_all_best, 'b-', label='GP Predictive Mean', linewidth=1.5, alpha=0.7)
plt.fill_between(t1, y1_pred_all_best - y1_std_all_best, y1_pred_all_best + y1_std_all_best,
                 color='blue', alpha=0.2, label='±1σ Predictive Uncertainty')

# Overlay test/train for this specific best split
plt.scatter(t1_train_best, y1_train_best, c='black', s=20, label=f'Training Points (Seed {best_k1_seed})', zorder=3)
plt.scatter(t1_test_best, y1_test_best, c='red', marker='x', s=50, label=f'Hidden Test Points (Seed {best_k1_seed})', zorder=4)

plt.xlabel("Time Index", fontsize=12)
plt.ylabel("Flux", fontsize=12)
plt.title(f"GPR Imputation on Kepler1 (Best Representative Split - Seed {best_k1_seed})", fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

    Generating visualization for Best Kepler 1 Seed: 11 (RMSE: 4.6718)



    
![png](test1_files/test1_14_1.png)
    



```python
# --- RANDOMIZED BLOCK MASKING EXPERIMENT ---
n_block_seeds = 20
block_size = int(len(t1_sub) * 0.2) # 20% continuous gap
max_start_idx = len(t1_sub) - block_size

gap_rmse_list, gap_mae_list, gap_1sigma_list, gap_2sigma_list = [], [], [], []

print("Running Randomized Block Masking Evaluation...")

for seed in range(n_block_seeds):
    rng = np.random.RandomState(seed)
    start_idx = rng.randint(0, max_start_idx)
    end_idx = start_idx + block_size

    # Create mask for the gap
    train_mask = np.ones(len(t1_sub), dtype=bool)
    train_mask[start_idx:end_idx] = False
    test_mask = ~train_mask

    t_train_gap, y_train_gap = t1_sub[train_mask], y1_sub[train_mask]
    t_test_gap, y_test_gap = t1_sub[test_mask], y1_sub[test_mask]

    # Fit & Predict
    gpr_gap = GaussianProcessRegressor(kernel=kernel1, alpha=1e-2, normalize_y=True, random_state=seed)
    gpr_gap.fit(t_train_gap.reshape(-1, 1), y_train_gap)
    y_pred_gap, y_std_gap = gpr_gap.predict(t_test_gap.reshape(-1, 1), return_std=True)

    # Evaluate
    gap_rmse_list.append(np.sqrt(mean_squared_error(y_test_gap, y_pred_gap)))
    gap_mae_list.append(mean_absolute_error(y_test_gap, y_pred_gap))
    gap_1sigma_list.append(np.mean((y_test_gap >= y_pred_gap - y_std_gap) & (y_test_gap <= y_pred_gap + y_std_gap)))
    gap_2sigma_list.append(np.mean((y_test_gap >= y_pred_gap - 2*y_std_gap) & (y_test_gap <= y_pred_gap + 2*y_std_gap)))

print("\n--- Block Masking (Continuous Gap) Aggregates ---")
print(f"Mean Gap RMSE: {np.mean(gap_rmse_list):.4f} ± {np.std(gap_rmse_list):.4f}")
print(f"Mean Gap MAE:  {np.mean(gap_mae_list):.4f} ± {np.std(gap_mae_list):.4f}")
print(f"Mean 1σ Coverage: {np.mean(gap_1sigma_list)*100:.2f}%")
print(f"Mean 2σ Coverage: {np.mean(gap_2sigma_list)*100:.2f}%")
```

    Running Randomized Block Masking Evaluation...
    
    --- Block Masking (Continuous Gap) Aggregates ---
    Mean Gap RMSE: 5.0891 ± 0.2717
    Mean Gap MAE:  4.1182 ± 0.2467
    Mean 1σ Coverage: 64.08%
    Mean 2σ Coverage: 95.83%


### Modeling Structured Periodicity with `Kepler2.dat` using _Quasi-Periodic_ Kernel


```python
from sklearn.gaussian_process.kernels import ExpSineSquared

# --- Kepler 2: Standard vs Quasi-Periodic Evaluation ---
# Step 1: Load and isolate valid data
flux2 = np.genfromtxt("datasets/Kepler2.dat", dtype=np.float32, missing_values="NA", filling_values=np.nan)
t2 = np.arange(len(flux2))

mask2 = ~np.isnan(flux2)
t2_valid = t2[mask2]
y2_valid = flux2[mask2]

# Subsample due to memory limitations
subset2 = 50
t2_sub = t2_valid[::subset2]
y2_sub = y2_valid[::subset2]

n_k2_seeds = 20

rbf_rmse_list, rbf_mae_list = [], []
qp_rmse_list, qp_mae_list = [], []

rbf_kernel2 = 1.0 * RBF(length_scale=100.0) + WhiteKernel(noise_level=1.0)
qp_kernel2 = 1.0 * ExpSineSquared(length_scale=100.0, periodicity=150.0) + 1.0 * RBF(length_scale=200.0) + WhiteKernel(noise_level=1.0)

print(f"Evaluating kernels on Kepler 2 across {n_k2_seeds} seeds...")

for seed in range(n_k2_seeds):
    t2_train, t2_test, y2_train, y2_test = train_test_split(t2_sub, y2_sub, test_size=0.2, random_state=seed)
    X2_train, X2_test = t2_train.reshape(-1, 1), t2_test.reshape(-1, 1)

    # RBF Baseline
    gpr2_rbf = GaussianProcessRegressor(kernel=rbf_kernel2, alpha=1e-2, normalize_y=True, random_state=seed)
    gpr2_rbf.fit(X2_train, y2_train)
    y_pred_rbf = gpr2_rbf.predict(X2_test)

    rbf_rmse_list.append(np.sqrt(mean_squared_error(y2_test, y_pred_rbf)))
    rbf_mae_list.append(mean_absolute_error(y2_test, y_pred_rbf))

    # Quasi-Periodic Target
    gpr2_qp = GaussianProcessRegressor(kernel=qp_kernel2, alpha=1e-2, normalize_y=True, random_state=seed)
    gpr2_qp.fit(X2_train, y2_train)
    y_pred_qp = gpr2_qp.predict(X2_test)

    qp_rmse_list.append(np.sqrt(mean_squared_error(y2_test, y_pred_qp)))
    qp_mae_list.append(mean_absolute_error(y2_test, y_pred_qp))

# --- Kepler 2 Summary Table ---
k2_benchmark_data = {
    "Kernel Method": ["RBF (Standard)", "Quasi-Periodic"],
    "RMSE Mean": [np.mean(rbf_rmse_list), np.mean(qp_rmse_list)],
    "RMSE Std": [np.std(rbf_rmse_list), np.std(qp_rmse_list)],
    "MAE Mean": [np.mean(rbf_mae_list), np.mean(qp_mae_list)],
    "MAE Std": [np.std(rbf_mae_list), np.std(qp_mae_list)]
}

k2_benchmark_df = pd.DataFrame(k2_benchmark_data)
print("\n--- Kepler 2 Kernel Robustness Comparison ---")
display.display(k2_benchmark_df.round(4))
```

    Evaluating kernels on Kepler 2 across 20 seeds...


    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(


    
    --- Kepler 2 Kernel Robustness Comparison ---


    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Kernel Method</th>
      <th>RMSE Mean</th>
      <th>RMSE Std</th>
      <th>MAE Mean</th>
      <th>MAE Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RBF (Standard)</td>
      <td>36.5371</td>
      <td>9.0953</td>
      <td>16.7843</td>
      <td>2.1005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Quasi-Periodic</td>
      <td>36.5492</td>
      <td>9.0948</td>
      <td>16.7911</td>
      <td>2.1034</td>
    </tr>
  </tbody>
</table>
</div>



```python
# --- Visualization of the Best Quasi-Periodic Model Split ---

# Step 1: Identify the best performing seed for the QP kernel
best_qp_idx = np.argmin(qp_rmse_list)
best_qp_seed = best_qp_idx # Since our seeds were range(n_k2_seeds)

print(f"Generating visualization for Best QP Seed: {best_qp_seed} (RMSE: {qp_rmse_list[best_qp_idx]:.4f})")

# Step 2: Recreate the train/test split for that specific best seed
t2_train_best, t2_test_best, y2_train_best, y2_test_best = train_test_split(
    t2_sub, y2_sub, test_size=0.2, random_state=best_qp_seed
)

X2_train_best = t2_train_best.reshape(-1, 1)

# Step 3: Retrain the QP model on the best split
gpr2_qp_best = GaussianProcessRegressor(kernel=qp_kernel2, alpha=1e-2, normalize_y=True, random_state=best_qp_seed)
gpr2_qp_best.fit(X2_train_best, y2_train_best)

# Step 4: Predict over the FULL time range to generate the smooth background band
X2_all = t2.reshape(-1, 1)
y2_pred_all_qp, y2_std_all_qp = gpr2_qp_best.predict(X2_all, return_std=True)

# Step 5: Publication-style Plotting
plt.figure(figsize=(14, 6))

# Plot full predictive band for the winning Quasi-Periodic model
plt.plot(t2, y2_pred_all_qp, 'b-', label='Quasi-Periodic GP Mean', alpha=0.7)
plt.fill_between(t2, y2_pred_all_qp - y2_std_all_qp, y2_pred_all_qp + y2_std_all_qp,
                 color='blue', alpha=0.2, label='±1σ Uncertainty (QP)')

# Overlay test/train for this specific best split
plt.scatter(t2_train_best, y2_train_best, c='black', s=10, label=f'Training Points (Seed {best_qp_seed})')
plt.scatter(t2_test_best, y2_test_best, c='red', marker='x', s=40, label=f'Hidden Test Points (Seed {best_qp_seed})')

plt.xlabel("Time Index", fontsize=12)
plt.ylabel("Flux", fontsize=12)
plt.title(f"Quasi-Periodic GPR Fit to Kepler2 (Best Representative Split - Seed {best_qp_seed})", fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

    Generating visualization for Best QP Seed: 7 (RMSE: 22.6351)


    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k1__constant_value is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:452: ConvergenceWarning: The optimal value found for dimension 0 of parameter k1__k1__k2__length_scale is close to the specified upper bound 100000.0. Increasing the bound and calling fit again may find a better value.
      warnings.warn(
    /Users/shyambanerjee/PycharmProjects/noisy_light_curve_simulation/.venv/lib/python3.9/site-packages/sklearn/gaussian_process/kernels.py:442: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__noise_level is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
      warnings.warn(



    
![png](test1_files/test1_18_2.png)
    

