"""
Gradient Descent Optimization for Battery Aging Diagnostics
Using Differential Voltage Analysis (DVA) with synthetic data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize
import os
import json
import time
from pathlib import Path

# Constants
s_half = 1e-12
s_cyc = 9e-7
step = 3
cscale = 1e3

# Create output directories
os.makedirs('images', exist_ok=True)
os.makedirs('results', exist_ok=True)

def deform(k, b, x):
    """Apply shrinkage (k) and shift (b) to capacity axis"""
    return x * k - b

def diff(X, Y, step):
    """Calculate numerical derivative dY/dX"""
    Y = np.array(Y)
    X = np.array(X)
    dY = Y[step:] - Y[:-step]
    dX = X[step:] - X[:-step]
    return dY / dX

def mesh(a, b, c):
    """Create common capacity mesh for interpolation"""
    data_lst = [a, b, c]
    x1 = max([min(i) for i in data_lst])
    x2 = min([max(i) for i in data_lst])
    x = []
    for i in data_lst:
        x += [j for j in i if j > x1 and j < x2]
    x = list(set(x))
    x = np.array(np.sort(x))
    return x, x1, x2

def interp_mesh(X, Y, z, s_num=s_half):
    """Spline interpolation for smoothing"""
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    if not all(X[i] < X[i + 1] for i in range(len(X) - 1)):
        raise ValueError("X must be strictly increasing")
    try:
        spl = splrep(X, Y, s=s_num)
        return splev(z, spl)
    except ValueError as e:
        print(f"Interpolation error: {e}")
        raise

def load_data():
    """Load synthetic battery data from CSV files"""
    data_dir = Path("synthetic_battery_data")
    
    print("Loading synthetic battery data...")
    
    # Load cathode half-cell
    cathode_discharge = pd.read_csv(data_dir / "cathode_halfcell_discharge.csv")
    cathode_charge = pd.read_csv(data_dir / "cathode_halfcell_charge.csv")
    
    # Load anode half-cell
    anode_charge = pd.read_csv(data_dir / "anode_halfcell_charge.csv")
    anode_discharge = pd.read_csv(data_dir / "anode_halfcell_discharge.csv")
    
    # Load full cell aging data
    fullcell_data = pd.read_csv(data_dir / "fullcell_aging_data.csv")
    
    print("[OK] Data loaded successfully")
    return cathode_discharge, cathode_charge, anode_charge, anode_discharge, fullcell_data

def optimize_cycle_GD(catx_D, caty_D, anox_D, anoy_D, catx_C, caty_C, anox_C, anoy_C,
                      cycx_D, cycy_D, cycx_C, cycy_C, cycle_num, run_idx=0):
    """
    Optimize k and b parameters using Gradient Descent for a single cycle
    
    Parameters:
    - kc, bc: cathode shrinkage and shift
    - ka, ba: anode shrinkage and shift
    - r: internal resistance
    """
    
    current_fast = 0.001001144  # Fast cycling current (C/3)
    current_slow = 0.00012      # Slow cycling current (C/25)
    mod_D = 1e-4
    mod_C = 1e-4
    loss_start = 3.2
    loss_end = 4.5
    
    # Parameter bounds
    pbounds = {
        'kc': (0.8, 1.2),
        'bc': (-1.5, 0),
        'ka': (0.9, 1.2),
        'ba': (-1.5, 0),
        'r': (0, 0.1)
    }
    
    def loss_DVQ(ka, kc, ba, bc, r, cycx, cycy, charge_type):
        """Calculate loss for differential voltage curve fitting"""
        if charge_type == 'Discharge':
            catx_def = deform(kc, bc, catx_D)
            anox_def = deform(ka, ba, anox_D)
            caty = caty_D
            anoy = anoy_D
        else:
            catx_def = deform(kc, bc, catx_C)
            anox_def = deform(ka, ba, anox_C)
            caty = caty_C
            anoy = anoy_C
        
        try:
            x, x1, x2 = mesh(cycx, catx_def, anox_def)
            caty_def = interp_mesh(catx_def, caty, x, s_num=s_half)
            anoy_def = interp_mesh(anox_def, anoy, x, s_num=s_half)
            cycy_def = interp_mesh(cycx, cycy, x, s_num=s_cyc)
            
            dVdQ_cat = diff(x, caty_def, step)
            dVdQ_ano = diff(x, anoy_def, step)
            dVdQ_cyc = diff(x, cycy_def, step)
            dVdQ_fit = dVdQ_cat - dVdQ_ano
            
            err_vec = dVdQ_fit - dVdQ_cyc
            loss = np.dot(err_vec, err_vec)
            return loss
        except:
            return 1e10
    
    def total_loss(params):
        """Total loss function combining discharge and charge, VQ and DVQ"""
        ka, ba, kc, bc, r = params
        
        # Discharge loss
        catx_def_D = deform(kc, bc, catx_D)
        anox_def_D = deform(ka, ba, anox_D)
        catx_def_C = deform(kc, bc, catx_C)
        anox_def_C = deform(ka, ba, anox_C)
        
        caty_normed_D = caty_D
        anoy_normed_D = anoy_D
        caty_normed_C = np.flip(caty_C)
        anoy_normed_C = np.flip(anoy_C)
        
        try:
            x_D, x1_D, x2_D = mesh(cycx_D, catx_def_D, anox_def_D)
            x_C, x1_C, x2_C = mesh(cycx_C, catx_def_C, anox_def_C)
            x_D = x_D[(x_D > loss_start) & (x_D < loss_end)]
            x_C = x_C[(x_C > loss_start) & (x_C < loss_end)]
            
            if len(x_C) == 0 or len(x_D) == 0:
                return 1e10
            
            caty_def_D = interp_mesh(catx_def_D, caty_normed_D, x_D, s_num=s_half)
            anoy_def_D = interp_mesh(anox_def_D, anoy_normed_D, x_D, s_num=s_half)
            cycy_def_D = interp_mesh(cycx_D, cycy_D, x_D, s_num=s_cyc)
            
            caty_def_C = interp_mesh(catx_def_C, caty_normed_C, x_C, s_num=s_half)
            anoy_def_C = interp_mesh(anox_def_C, anoy_normed_C, x_C, s_num=s_half)
            cycy_def_C = interp_mesh(cycx_C, cycy_C, x_C, s_num=s_cyc)
            
            VQ_fit_D = caty_def_D - anoy_def_D - r * current_slow
            err_vec_D = VQ_fit_D - cycy_def_D
            VQ_fit_C = caty_def_C - anoy_def_C + r * current_slow
            err_vec_C = VQ_fit_C - cycy_def_C
            
            loss_D = (np.dot(err_vec_D, err_vec_D)) / len(err_vec_D) + \
                     loss_DVQ(ka, kc, ba, bc, r, cycx_D, cycy_D, "Discharge") * mod_D
            loss_C = (np.dot(err_vec_C, err_vec_C)) / len(err_vec_C) + \
                     loss_DVQ(ka, kc, ba, bc, r, cycx_C, cycy_C, "Charge") * mod_C
            
            return loss_D + loss_C
        except:
            return 1e10
    
    # Random initialization
    initial_guess = [
        np.random.uniform(*pbounds['ka']),
        np.random.uniform(*pbounds['ba']),
        np.random.uniform(*pbounds['kc']),
        np.random.uniform(*pbounds['bc']),
        np.random.uniform(*pbounds['r'])
    ]
    
    # Optimize using L-BFGS-B
    param_order = ['ka', 'ba', 'kc', 'bc', 'r']
    bounds = [pbounds[key] for key in param_order]
    
    start_time = time.time()
    result = minimize(total_loss, initial_guess, method='L-BFGS-B', bounds=bounds)
    end_time = time.time()
    
    ka, ba, kc, bc, r = result.x
    final_loss = result.fun
    optimization_time = end_time - start_time
    
    return {
        'ka': ka,
        'ba': ba,
        'kc': kc,
        'bc': bc,
        'r': r,
        'loss': final_loss,
        'success': result.success,
        'time': optimization_time,
        'run_idx': run_idx
    }

def run_gradient_descent():
    """Main function to run Gradient Descent optimization on all cycles"""
    print("\n" + "="*70)
    print("GRADIENT DESCENT OPTIMIZATION FOR BATTERY AGING DIAGNOSTICS")
    print("="*70 + "\n")
    
    # Load data
    cathode_discharge, cathode_charge, anode_charge, anode_discharge, fullcell_data = load_data()
    
    # Prepare half-cell data
    catx_D = cathode_discharge['Amp_hr'].values * cscale
    caty_D = cathode_discharge['Volts'].values
    catx_C = cathode_charge['Amp_hr'].values * cscale
    caty_C = cathode_charge['Volts'].values
    
    anox_D = anode_discharge['Amp_hr'].values * cscale
    anoy_D = anode_discharge['Volts'].values
    anox_C = anode_charge['Amp_hr'].values * cscale
    anoy_C = anode_charge['Volts'].values
    
    # Get cycle list
    cycle_list = sorted(fullcell_data['Cycle_Number'].unique())
    print(f"Cycles to optimize: {cycle_list}\n")
    
    # Run multiple iterations for each cycle
    num_iterations = 9  # As in original paper
    all_results = []
    
    for cycle_num in cycle_list:
        print(f"\n{'='*70}")
        print(f"CYCLE {cycle_num}")
        print(f"{'='*70}")
        
        # Get full cell data for this cycle
        cycle_discharge = fullcell_data[
            (fullcell_data['Cycle_Number'] == cycle_num) &
            (fullcell_data['Type'] == 'Discharge')
        ]
        cycle_charge = fullcell_data[
            (fullcell_data['Cycle_Number'] == cycle_num) &
            (fullcell_data['Type'] == 'Charge')
        ]
        
        cycx_D = cycle_discharge['Amp_hr_actual'].values * cscale
        cycy_D = cycle_discharge['Volts'].values
        cycx_C = cycle_charge['Amp_hr_actual'].values * cscale
        cycy_C = cycle_charge['Volts'].values
        
        cycle_results = []
        
        for i in range(num_iterations):
            print(f"  Run {i+1}/{num_iterations}...", end=' ')
            
            result = optimize_cycle_GD(
                catx_D, caty_D, anox_D, anoy_D,
                catx_C, caty_C, anox_C, anoy_C,
                cycx_D, cycy_D, cycx_C, cycy_C,
                cycle_num, run_idx=i
            )
            
            result['cycle'] = cycle_num
            cycle_results.append(result)
            all_results.append(result)
            
            status = "[OK]" if result['success'] else "[FAIL]"
            print(f"{status} Loss: {result['loss']:.6f}, Time: {result['time']:.2f}s")
        
        # Find best result for this cycle
        valid_results = [r for r in cycle_results if r['success']]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['loss'])
            print(f"\n  Best result: Run {best_result['run_idx']+1}")
            print(f"    kc={best_result['kc']:.4f}, bc={best_result['bc']:.4f}")
            print(f"    ka={best_result['ka']:.4f}, ba={best_result['ba']:.4f}")
            print(f"    r={best_result['r']:.4f}")
            print(f"    Loss={best_result['loss']:.6f}")
            print(f"    LAM_cathode={(1-best_result['kc'])*100:.2f}%")
            print(f"    LAM_anode={(1-best_result['ka'])*100:.2f}%")
            print(f"    LLI={max(best_result['ba'], best_result['bc']):.4f} mAh")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/GD_all_results.csv', index=False)
    
    # Get best results for each cycle
    best_results = []
    for cycle_num in cycle_list:
        cycle_res = results_df[results_df['cycle'] == cycle_num]
        valid_res = cycle_res[cycle_res['success'] == True]
        if len(valid_res) > 0:
            best_idx = valid_res['loss'].idxmin()
            best_results.append(valid_res.loc[best_idx])
    
    best_df = pd.DataFrame(best_results)
    best_df.to_csv('results/GD_best_results.csv', index=False)
    
    # Calculate statistics
    success_rate = (results_df['success'].sum() / len(results_df)) * 100
    avg_time = results_df['time'].mean()
    
    print(f"\n{'='*70}")
    print("GRADIENT DESCENT RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total runs: {len(results_df)}")
    print(f"Successful convergences: {results_df['success'].sum()} ({success_rate:.1f}%)")
    print(f"Average optimization time: {avg_time:.2f}s")
    print(f"\nResults saved to:")
    print(f"  - results/GD_all_results.csv")
    print(f"  - results/GD_best_results.csv")
    
    return best_df, results_df

if __name__ == "__main__":
    best_results, all_results = run_gradient_descent()
    print("\n[OK] Gradient Descent optimization completed!\n")
