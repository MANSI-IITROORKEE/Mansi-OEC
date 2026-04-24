"""
Synthetic Battery Aging Data Generator for DVA Analysis
Generates realistic lithium-ion battery voltage-capacity curves
with aging effects (LAM and LLI) for NMC532-Graphite cells
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os

# Set random seed for reproducibility
np.random.seed(42)

# Constants
CSCALE = 1e3  # mAh scale
NUM_CYCLES = [3, 54, 105, 156, 207, 258, 309]  # Cycle numbers to generate
NOISE_LEVEL = 0.001  # Voltage noise level

def generate_cathode_halfcell(Q_max=4.2, num_points=200):
    """
    Generate NMC532 cathode half-cell voltage curve (vs Li/Li+)
    Voltage range: 2.5 - 4.3V
    """
    # Capacity values (normalized)
    Q = np.linspace(0, Q_max, num_points)
    
    # NMC cathode voltage profile with characteristic shape
    # Multiple regions representing phase transitions
    V = np.zeros(num_points)
    for i, q in enumerate(Q):
        q_norm = q / Q_max
        # Three-region voltage profile typical of NMC
        if q_norm < 0.15:
            V[i] = 4.3 - 0.5 * q_norm
        elif q_norm < 0.85:
            V[i] = 4.2 - 0.15 * (q_norm - 0.15) - 0.3 * np.sin(8 * np.pi * (q_norm - 0.15))
        else:
            V[i] = 3.7 - 1.0 * (q_norm - 0.85)
    
    # Add small peaks for phase transitions
    peak1 = 0.25
    peak2 = 0.50
    peak3 = 0.75
    V += 0.05 * np.exp(-50 * (Q/Q_max - peak1)**2)
    V += 0.07 * np.exp(-50 * (Q/Q_max - peak2)**2)
    V += 0.06 * np.exp(-50 * (Q/Q_max - peak3)**2)
    
    # Add measurement noise
    V += np.random.normal(0, NOISE_LEVEL, num_points)
    
    # Clip to physical range
    V = np.clip(V, 2.5, 4.3)
    
    return Q * CSCALE, V

def generate_anode_halfcell(Q_max=5.4, num_points=200):
    """
    Generate Graphite anode half-cell voltage curve (vs Li/Li+)
    Voltage range: 0.01 - 1.5V
    """
    Q = np.linspace(0, Q_max, num_points)
    
    # Graphite anode voltage profile with characteristic staging
    V = np.zeros(num_points)
    for i, q in enumerate(Q):
        q_norm = q / Q_max
        # Graphite staging behavior with plateaus
        V[i] = 0.05 + 0.15 * q_norm + 0.5 * np.exp(-10 * q_norm)
        
        # Add staging plateaus
        if 0.1 < q_norm < 0.3:
            V[i] += 0.05
        if 0.4 < q_norm < 0.6:
            V[i] += 0.08
        if 0.7 < q_norm < 0.9:
            V[i] += 0.06
    
    # Add staging peaks in differential voltage
    peak_positions = [0.15, 0.35, 0.55, 0.75]
    for peak_pos in peak_positions:
        V += 0.03 * np.exp(-100 * (Q/Q_max - peak_pos)**2)
    
    # Add measurement noise
    V += np.random.normal(0, NOISE_LEVEL, num_points)
    
    # Clip to physical range
    V = np.clip(V, 0.01, 1.5)
    
    return Q * CSCALE, V

def apply_aging_effects(Q, V, cycle_num, electrode_type='cathode'):
    """
    Apply aging effects (LAM and LLI) to voltage curves
    
    Parameters:
    - LAM (Loss of Active Material): reduces capacity -> k parameter (shrinkage)
    - LLI (Loss of Lithium Inventory): shifts curve -> b parameter (shift)
    """
    if cycle_num <= 3:
        return Q, V
    
    # Progressive aging: increases with cycle number
    aging_factor = (cycle_num - 3) / 306  # Normalized aging (0 to 1)
    
    if electrode_type == 'cathode':
        # Cathode experiences more LAM than anode (from paper: ~6% at end)
        k = 1.0 - 0.06 * aging_factor  # Shrinkage factor
        b = 0.6 * aging_factor * CSCALE  # Shift in mAh
    else:  # anode
        # Anode experiences less LAM (from paper: ~2.5% at end)
        k = 1.0 - 0.025 * aging_factor
        b = 0.6 * aging_factor * CSCALE
    
    # Apply deformation: Q_new = k * Q - b
    Q_aged = k * Q - b
    
    return Q_aged, V

def generate_fullcell_from_halfcells(Q_cathode, V_cathode, Q_anode, V_anode, 
                                     cycle_num, num_points=200):
    """
    Generate full cell voltage curve from half-cell curves
    V_fullcell = V_cathode - V_anode
    """
    # Apply aging to both electrodes
    Q_cat_aged, V_cat = apply_aging_effects(Q_cathode, V_cathode, cycle_num, 'cathode')
    Q_ano_aged, V_ano = apply_aging_effects(Q_anode, V_anode, cycle_num, 'anode')
    
    # Find common capacity range
    Q_min = max(Q_cat_aged.min(), Q_ano_aged.min())
    Q_max = min(Q_cat_aged.max(), Q_ano_aged.max())
    
    # Create common capacity axis
    Q_common = np.linspace(Q_min, Q_max, num_points)
    
    # Interpolate both half-cells to common axis
    f_cat = interp1d(Q_cat_aged, V_cat, kind='cubic', fill_value='extrapolate')
    f_ano = interp1d(Q_ano_aged, V_ano, kind='cubic', fill_value='extrapolate')
    
    V_cat_interp = f_cat(Q_common)
    V_ano_interp = f_ano(Q_common)
    
    # Full cell voltage
    V_full = V_cat_interp - V_ano_interp
    
    # Add resistance effect (IR drop)
    current = 0.00012  # A (C/25 rate)
    resistance = 0.04 + 0.02 * (cycle_num - 3) / 306  # Increases with aging
    V_full -= resistance * current
    
    # Add noise
    V_full += np.random.normal(0, NOISE_LEVEL, num_points)
    
    # Clip to typical full cell range
    V_full = np.clip(V_full, 3.0, 4.1)
    
    return Q_common, V_full

def save_to_csv():
    """Generate and save all synthetic data to CSV files"""
    
    # Create data directory
    data_dir = "synthetic_battery_data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating synthetic battery aging data...")
    
    # 1. Generate pristine half-cells
    print("\n1. Generating cathode half-cell data...")
    Q_cathode, V_cathode = generate_cathode_halfcell()
    
    # Save cathode discharge (reversed for discharge)
    cathode_df = pd.DataFrame({
        'Amp_hr': Q_cathode / CSCALE,
        'Volts': V_cathode,
        'Cycle': 'Discharge 2'
    })
    cathode_df.to_csv(f"{data_dir}/cathode_halfcell_discharge.csv", index=False)
    
    # Save cathode charge
    cathode_charge_df = pd.DataFrame({
        'Amp_hr': np.flip(Q_cathode) / CSCALE,
        'Volts': np.flip(V_cathode),
        'Cycle': 'Charge 2'
    })
    cathode_charge_df.to_csv(f"{data_dir}/cathode_halfcell_charge.csv", index=False)
    
    print(f"   Saved: cathode_halfcell_discharge.csv ({len(cathode_df)} points)")
    print(f"   Saved: cathode_halfcell_charge.csv ({len(cathode_charge_df)} points)")
    
    # 2. Generate anode half-cells
    print("\n2. Generating anode half-cell data...")
    Q_anode, V_anode = generate_anode_halfcell()
    
    # Save anode charge
    anode_df = pd.DataFrame({
        'Amp_hr': Q_anode / CSCALE,
        'Volts': V_anode,
        'Cycle': 'Charge 5'
    })
    anode_df.to_csv(f"{data_dir}/anode_halfcell_charge.csv", index=False)
    
    # Save anode discharge
    anode_discharge_df = pd.DataFrame({
        'Amp_hr': np.flip(Q_anode) / CSCALE,
        'Volts': np.flip(V_anode),
        'Cycle': 'Discharge 5'
    })
    anode_discharge_df.to_csv(f"{data_dir}/anode_halfcell_discharge.csv", index=False)
    
    print(f"   Saved: anode_halfcell_charge.csv ({len(anode_df)} points)")
    print(f"   Saved: anode_halfcell_discharge.csv ({len(anode_discharge_df)} points)")
    
    # 3. Generate full cell data at different cycles
    print("\n3. Generating full cell aging data...")
    all_fullcell_data = []
    
    for cycle_num in NUM_CYCLES:
        # Generate discharge curve
        Q_full_discharge, V_full_discharge = generate_fullcell_from_halfcells(
            Q_cathode, V_cathode, Q_anode, V_anode, cycle_num
        )
        
        # Reverse for discharge direction
        Q_full_discharge = np.flip(Q_full_discharge)
        V_full_discharge = np.flip(V_full_discharge)
        
        discharge_df = pd.DataFrame({
            'Cycle_Number': cycle_num,
            'Amp_hr_actual': Q_full_discharge / CSCALE,
            'Volts': V_full_discharge,
            'Type': 'Discharge'
        })
        
        # Generate charge curve (reverse direction)
        Q_full_charge = np.flip(Q_full_discharge)
        V_full_charge = np.flip(V_full_discharge)
        
        charge_df = pd.DataFrame({
            'Cycle_Number': cycle_num,
            'Amp_hr_actual': Q_full_charge / CSCALE,
            'Volts': V_full_charge,
            'Type': 'Charge'
        })
        
        all_fullcell_data.append(discharge_df)
        all_fullcell_data.append(charge_df)
        
        print(f"   Generated cycle {cycle_num}: Discharge ({len(discharge_df)} points), Charge ({len(charge_df)} points)")
    
    # Combine and save all full cell data
    fullcell_df = pd.concat(all_fullcell_data, ignore_index=True)
    fullcell_df.to_csv(f"{data_dir}/fullcell_aging_data.csv", index=False)
    print(f"\n   Saved: fullcell_aging_data.csv ({len(fullcell_df)} total points)")
    
    # 4. Generate summary statistics
    print("\n4. Generating dataset summary...")
    summary_data = []
    
    for cycle_num in NUM_CYCLES:
        cycle_data = fullcell_df[
            (fullcell_df['Cycle_Number'] == cycle_num) & 
            (fullcell_df['Type'] == 'Discharge')
        ]
        
        capacity = cycle_data['Amp_hr_actual'].max()
        mean_voltage = cycle_data['Volts'].mean()
        # Use trapezoid for newer numpy versions
        try:
            energy = np.trapezoid(cycle_data['Volts'], cycle_data['Amp_hr_actual'])
        except AttributeError:
            energy = np.trapz(cycle_data['Volts'], cycle_data['Amp_hr_actual'])
        
        summary_data.append({
            'Cycle_Number': cycle_num,
            'Capacity_mAh': capacity * 1000,
            'Mean_Voltage_V': mean_voltage,
            'Energy_Wh': energy,
            'Capacity_Fade_%': (1 - capacity / (fullcell_df[
                fullcell_df['Cycle_Number'] == 3]['Amp_hr_actual'].max())) * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{data_dir}/summary_statistics.csv", index=False)
    print(f"   Saved: summary_statistics.csv")
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE!")
    print("="*60)
    print(f"\nDataset location: ./{data_dir}/")
    print(f"\nGenerated files:")
    print(f"  1. cathode_halfcell_discharge.csv")
    print(f"  2. cathode_halfcell_charge.csv")
    print(f"  3. anode_halfcell_charge.csv")
    print(f"  4. anode_halfcell_discharge.csv")
    print(f"  5. fullcell_aging_data.csv (Main dataset)")
    print(f"  6. summary_statistics.csv")
    print(f"\nTotal cycles: {len(NUM_CYCLES)}")
    print(f"Cycle numbers: {NUM_CYCLES}")
    print("\nDataset characteristics:")
    print(f"  - Cathode: NMC532 (LAM ~6% at end of life)")
    print(f"  - Anode: Graphite (LAM ~2.5% at end of life)")
    print(f"  - LLI: ~0.6 mAh at end of life")
    print(f"  - Capacity fade: ~10% over 309 cycles")
    print(f"  - Voltage range: 3.0-4.1V (full cell)")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("\n")

if __name__ == "__main__":
    save_to_csv()
