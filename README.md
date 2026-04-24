# Battery Aging Diagnostics - Setup and Execution Guide

This project implements machine learning optimization methods (Gradient Descent and Bayesian Optimization) for lithium-ion battery aging diagnostics based on research from the Journal of Power Sources (2025).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Setup Virtual Environment](#step-1-setup-virtual-environment)
3. [Step 2: Install Dependencies](#step-2-install-dependencies)
4. [Step 3: Generate Synthetic Dataset](#step-3-generate-synthetic-dataset)
5. [Step 4: Run Gradient Descent Model](#step-4-run-gradient-descent-model)
6. [Step 5: Run Bayesian Optimization Model](#step-5-run-bayesian-optimization-model)
7. [Step 6: Generate Metrics and Visualizations](#step-6-generate-metrics-and-visualizations)
8. [Step 7: View Results](#step-7-view-results)
9. [Troubleshooting](#troubleshooting)
10. [Project Structure](#project-structure)

---

## Prerequisites

- Python 3.8 or higher installed on your system
- Terminal/Command Prompt access
- Basic understanding of command line operations

---

## Step 1: Setup Virtual Environment

### On Windows (PowerShell or Command Prompt):

```powershell
# Navigate to project directory : Example
cd "D:\python programs\Data Science\Mansi-OEC"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### On macOS/Linux:

```bash
# Navigate to project directory
cd "/path/to/Mansi-OEC"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Verify activation:** You should see `(venv)` prefix in your terminal prompt.

---

## Step 2: Install Dependencies

After activating the virtual environment, install all required packages:

```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

**Expected output:** Installation progress for numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, and bayesian-optimization.

**Verify installation:**

```bash
pip list
```

You should see all installed packages with their versions.

---

## Step 3: Generate Synthetic Dataset

Generate the synthetic battery aging dataset:

```bash
python generate_synthetic_data.py
```

**Expected output:**
- Console messages showing data generation progress
- Creation of `synthetic_battery_data/` folder
- 6 CSV files created (cathode/anode half-cells, full-cell aging data, summary statistics)

**Verify:**

```bash
# Check if folder was created (Windows)
dir synthetic_battery_data

# Check if folder was created (macOS/Linux)
ls synthetic_battery_data/
```

You should see 6 CSV files.

---

## Step 4: Run Gradient Descent Model

Run the Gradient Descent optimization model:

```bash
python DVA_GradientDescent.py
```

**Expected duration:** ~2-3 seconds

**Expected output:**
- Console messages showing optimization progress for 7 cycles
- 63 total optimization runs (9 per cycle)
- Creation of `results/` folder
- `GD_all_results.csv` and `GD_best_results.csv` files created

**Verify:**

```bash
# Check results folder (Windows)
dir results

# Check results folder (macOS/Linux)
ls results/
```

You should see GD result CSV files.

---

## Step 5: Run Bayesian Optimization Model

Run the Bayesian Optimization model:

```bash
python DVA_BayesianOptimization.py
```

**Expected duration:** ~8-10 minutes (significantly slower than GD)

**Expected output:**
- Console messages showing BO progress for 7 cycles
- 63 total optimization runs with 3 acquisition functions (EI, UCB, POI)
- `BO_all_results.csv` and `BO_best_results.csv` created in `results/` folder

**Note:** This step takes longer due to Bayesian Optimization's computational complexity.

---

## Step 6: Generate Metrics and Visualizations

Calculate performance metrics and generate comparison graphs:

```bash
python metrics_and_visualization.py
```

**Expected output:**
- Error metrics (MAE, RMSE) printed to console
- Creation of `images/` folder
- 3 PNG files generated:
  - `capacity_energy_fade.png` - Battery degradation over cycles
  - `aging_comparison.png` - LAM and LLI comparison (GD vs BO)
  - `performance_comparison.png` - Algorithm performance metrics
- `error_metrics.json` and `combined_aging_metrics.csv` created in `results/` folder

**Verify:**

```bash
# Check images folder (Windows)
dir images

# Check images folder (macOS/Linux)
ls images/
```

You should see 3 PNG files.

---

## Step 7: View Results

### View the Comprehensive Report

Open `report.md` in any markdown viewer or text editor to read the complete 4,500+ word technical report.

### View Generated Graphs

Navigate to the `images/` folder and open the PNG files:

1. **capacity_energy_fade.png** - Shows battery capacity decline from 4200 to 3348 mAh over 309 cycles
2. **aging_comparison.png** - Compares predicted vs expected aging mechanisms (LAM and LLI)
3. **performance_comparison.png** - Compares GD and BO performance metrics

### View Raw Data

Open CSV files in `results/` folder:

- `GD_all_results.csv` - All 63 Gradient Descent runs
- `GD_best_results.csv` - Best GD result per cycle
- `BO_all_results.csv` - All 63 Bayesian Optimization runs
- `BO_best_results.csv` - Best BO result per cycle
- `combined_aging_metrics.csv` - Merged aging analysis
- `error_metrics.json` - MAE and RMSE metrics

---

## Troubleshooting

### Issue: "python: command not found"

**Solution:** Try `python3` instead of `python`:

```bash
python3 generate_synthetic_data.py
```

### Issue: Virtual environment not activating

**Windows:** Make sure to run PowerShell as Administrator or change execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:** Ensure you have correct permissions:

```bash
chmod +x venv/bin/activate
```

### Issue: Module not found errors

**Solution:** Ensure virtual environment is activated and reinstall dependencies:

```bash
# Check if venv is active (should see (venv) prefix)
pip install -r requirements.txt --force-reinstall
```

### Issue: "No such file or directory"

**Solution:** Verify you're in the correct project directory:

```bash
# Windows
cd "D:\python programs\Data Science\Mansi-OEC"

# macOS/Linux
pwd  # Should show path to Mansi-OEC folder
```

### Issue: Slow Bayesian Optimization

**Expected behavior:** BO is computationally intensive and takes 8-10 minutes. This is normal.

**Optional:** You can skip BO and only run GD for quick testing:

```bash
python DVA_GradientDescent.py
python metrics_and_visualization.py
```

Note: Metrics script will only compare GD results if BO hasn't been run yet.

---

## Project Structure

```
Mansi-OEC/
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file - Setup guide
├── report.md                     # Comprehensive technical report
├── Group-10.pdf                  # Original research paper
│
├── generate_synthetic_data.py   # Dataset generator script
├── DVA_GradientDescent.py       # Gradient Descent model
├── DVA_BayesianOptimization.py  # Bayesian Optimization model
├── metrics_and_visualization.py # Metrics calculator
│
├── venv/                         # Virtual environment (created in Step 1)
│
├── synthetic_battery_data/       # Generated datasets (created in Step 3)
│   ├── cathode_halfcell_discharge.csv
│   ├── cathode_halfcell_charge.csv
│   ├── anode_halfcell_charge.csv
│   ├── anode_halfcell_discharge.csv
│   ├── fullcell_aging_data.csv
│   └── summary_statistics.csv
│
├── results/                      # Optimization results (created in Steps 4-6)
│   ├── GD_all_results.csv
│   ├── GD_best_results.csv
│   ├── BO_all_results.csv
│   ├── BO_best_results.csv
│   ├── combined_aging_metrics.csv
│   └── error_metrics.json
│
└── images/                       # Visualizations (created in Step 6)
    ├── capacity_energy_fade.png
    ├── aging_comparison.png
    └── performance_comparison.png
```

---

## Quick Start (All Commands)

For a complete run-through, copy and paste these commands sequentially:

### Windows:

```powershell
# Setup
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Run pipeline
python generate_synthetic_data.py
python DVA_GradientDescent.py
python DVA_BayesianOptimization.py
python metrics_and_visualization.py

# View results
dir synthetic_battery_data
dir results
dir images
```

### macOS/Linux:

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run pipeline
python generate_synthetic_data.py
python DVA_GradientDescent.py
python DVA_BayesianOptimization.py
python metrics_and_visualization.py

# View results
ls synthetic_battery_data/
ls results/
ls images/
```

---

## Key Results Summary

### Performance Comparison

| Metric | Gradient Descent | Bayesian Optimization |
|--------|------------------|----------------------|
| **Speed** | 0.02s per optimization | 7.96s per optimization |
| **Success Rate** | 100% | 100% |
| **MAE (LAM Cathode)** | 11.89% | 4.95% |
| **MAE (LLI)** | 0.817 mAh | 0.383 mAh |
| **Best For** | Quick screening | Accurate verification |

### Recommended Workflow

1. Use **Gradient Descent** first for fast initial screening
2. Use **Bayesian Optimization** for verification and higher accuracy
3. If results agree → high confidence in diagnosis
4. If results disagree → complex optimization landscape, investigate further

---

## Dependencies Installed

From `requirements.txt`:

- **numpy** (≥2.4.0) - Numerical computing
- **pandas** (≥3.0.0) - Data manipulation
- **scipy** (≥1.17.0) - Scientific computing and optimization
- **matplotlib** (≥3.10.0) - Plotting and visualization
- **seaborn** (≥0.13.0) - Statistical visualization
- **scikit-learn** (≥1.8.0) - Machine learning utilities
- **bayesian-optimization** (≥3.2.0) - Bayesian optimization framework

---

## Additional Documentation

- **report.md** - Comprehensive technical report (4,500+ words) with methodology, results, and analysis
- **EXECUTION_SUMMARY.md** - Detailed execution logs from successful test runs
- **README_SUMMARY.md** - Project overview and achievements summary
- **Group-10.pdf** - Original research paper this project is based on

---

## Support

If you encounter any issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Verify all steps were completed in order
3. Ensure virtual environment is activated (look for `(venv)` prefix)
4. Review console output for specific error messages

---

## Citation

This project implements methods from:

Zhao, S., Howey, D. A., et al. (2025). "Benchmarking optimization methods for materials research: Gradient descent and Bayesian optimization for lithium-ion battery aging diagnostics." *Journal of Power Sources*, Volume 625, Article 235533.

---

**Project Status:** ✅ COMPLETE AND TESTED

**Last Updated:** April 25, 2026
