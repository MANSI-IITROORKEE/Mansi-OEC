"""
Performance Metrics Calculation and Visualization
for Battery Aging Diagnostics using GD and BO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directories
os.makedirs('images', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_results():
    """Load optimization results from both methods"""
    results_dir = Path('results')
    
    # Load GD results
    try:
        gd_best = pd.read_csv(results_dir / 'GD_best_results.csv')
        gd_all = pd.read_csv(results_dir / 'GD_all_results.csv')
        print("[OK] Loaded Gradient Descent results")
    except FileNotFoundError:
        print("[WARNING] GD results not found")
        gd_best, gd_all = None, None
    
    # Load BO results
    try:
        bo_best = pd.read_csv(results_dir / 'BO_best_results.csv')
        bo_all = pd.read_csv(results_dir / 'BO_all_results.csv')
        print("[OK] Loaded Bayesian Optimization results")
    except FileNotFoundError:
        print("[WARNING] BO results not found")
        bo_best, bo_all = None, None
    
    return gd_best, gd_all, bo_best, bo_all

def calculate_aging_metrics(results_df, method_name):
    """Calculate LAM and LLI from optimization results"""
    if results_df is None or len(results_df) == 0:
        return None
    
    metrics = []
    for _, row in results_df.iterrows():
        cycle = row['cycle']
        
        # Calculate LAM (Loss of Active Material)
        LAM_cathode = (1 - row['kc']) * 100  # Percentage
        LAM_anode = (1 - row['ka']) * 100    # Percentage
        
        # Calculate LLI (Loss of Lithium Inventory) - max of shifts
        LLI = max(abs(row['ba']), abs(row['bc']))
        
        metrics.append({
            'Cycle': cycle,
            'Method': method_name,
            'LAM_Cathode_%': LAM_cathode,
            'LAM_Anode_%': LAM_anode,
            'Total_LAM_%': LAM_cathode + LAM_anode,
            'LLI_mAh': LLI,
            'Resistance_Ohm': row['r'],
            'Loss': row['loss'],
            'Time_s': row['time']
        })
    
    return pd.DataFrame(metrics)

def plot_aging_comparison(gd_metrics, bo_metrics):
    """Plot LAM and LLI comparison between GD and BO"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Battery Aging Diagnostics: GD vs BO Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: LAM Cathode
    ax1 = axes[0, 0]
    if gd_metrics is not None:
        ax1.plot(gd_metrics['Cycle'], gd_metrics['LAM_Cathode_%'], 'o-', 
                label='GD', color='#2E86AB', linewidth=2, markersize=8)
    if bo_metrics is not None:
        ax1.plot(bo_metrics['Cycle'], bo_metrics['LAM_Cathode_%'], 's-', 
                label='BO', color='#A23B72', linewidth=2, markersize=8)
    
    # Add expected line (6% at end)
    if gd_metrics is not None or bo_metrics is not None:
        cycles = gd_metrics['Cycle'].values if gd_metrics is not None else bo_metrics['Cycle'].values
        expected_lam_cathode = [(c - 3) / 306 * 6 for c in cycles]
        ax1.plot(cycles, expected_lam_cathode, '--', label='Expected (6%)', 
                color='gray', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Cycle Number', fontweight='bold')
    ax1.set_ylabel('LAM Cathode (%)', fontweight='bold')
    ax1.set_title('Loss of Active Material - Cathode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: LAM Anode
    ax2 = axes[0, 1]
    if gd_metrics is not None:
        ax2.plot(gd_metrics['Cycle'], gd_metrics['LAM_Anode_%'], 'o-', 
                label='GD', color='#2E86AB', linewidth=2, markersize=8)
    if bo_metrics is not None:
        ax2.plot(bo_metrics['Cycle'], bo_metrics['LAM_Anode_%'], 's-', 
                label='BO', color='#A23B72', linewidth=2, markersize=8)
    
    # Add expected line (2.5% at end)
    if gd_metrics is not None or bo_metrics is not None:
        expected_lam_anode = [(c - 3) / 306 * 2.5 for c in cycles]
        ax2.plot(cycles, expected_lam_anode, '--', label='Expected (2.5%)', 
                color='gray', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Cycle Number', fontweight='bold')
    ax2.set_ylabel('LAM Anode (%)', fontweight='bold')
    ax2.set_title('Loss of Active Material - Anode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: LLI
    ax3 = axes[1, 0]
    if gd_metrics is not None:
        ax3.plot(gd_metrics['Cycle'], gd_metrics['LLI_mAh'], 'o-', 
                label='GD', color='#2E86AB', linewidth=2, markersize=8)
    if bo_metrics is not None:
        ax3.plot(bo_metrics['Cycle'], bo_metrics['LLI_mAh'], 's-', 
                label='BO', color='#A23B72', linewidth=2, markersize=8)
    
    # Add expected line (0.6 mAh at end)
    if gd_metrics is not None or bo_metrics is not None:
        expected_lli = [(c - 3) / 306 * 0.6 for c in cycles]
        ax3.plot(cycles, expected_lli, '--', label='Expected (0.6 mAh)', 
                color='gray', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Cycle Number', fontweight='bold')
    ax3.set_ylabel('LLI (mAh)', fontweight='bold')
    ax3.set_title('Loss of Lithium Inventory')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Resistance
    ax4 = axes[1, 1]
    if gd_metrics is not None:
        ax4.plot(gd_metrics['Cycle'], gd_metrics['Resistance_Ohm'], 'o-', 
                label='GD', color='#2E86AB', linewidth=2, markersize=8)
    if bo_metrics is not None:
        ax4.plot(bo_metrics['Cycle'], bo_metrics['Resistance_Ohm'], 's-', 
                label='BO', color='#A23B72', linewidth=2, markersize=8)
    
    ax4.set_xlabel('Cycle Number', fontweight='bold')
    ax4.set_ylabel('Resistance (Ω)', fontweight='bold')
    ax4.set_title('Internal Resistance Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/aging_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: images/aging_comparison.png")
    plt.close()

def plot_performance_comparison(gd_all, bo_all):
    """Plot performance metrics comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Optimization Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss Distribution
    ax1 = axes[0]
    if gd_all is not None and len(gd_all) > 0:
        # Filter out error values
        gd_valid = gd_all[gd_all['loss'] < 1e9]
        if len(gd_valid) > 0:
            ax1.hist(np.log10(gd_valid['loss']), bins=20, alpha=0.6, 
                    label=f'GD (n={len(gd_valid)})', color='#2E86AB', edgecolor='black')
    
    if bo_all is not None and len(bo_all) > 0:
        bo_valid = bo_all[bo_all['loss'] < 1e9]
        if len(bo_valid) > 0:
            ax1.hist(np.log10(bo_valid['loss']), bins=20, alpha=0.6, 
                    label=f'BO (n={len(bo_valid)})', color='#A23B72', edgecolor='black')
    
    ax1.set_xlabel('Log10(Loss)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Loss Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Computation Time
    ax2 = axes[1]
    methods = []
    times = []
    
    if gd_all is not None and len(gd_all) > 0:
        methods.append('Gradient\nDescent')
        times.append(gd_all['time'].mean())
    
    if bo_all is not None and len(bo_all) > 0:
        methods.append('Bayesian\nOptimization')
        times.append(bo_all['time'].mean())
    
    if len(methods) > 0:
        bars = ax2.bar(methods, times, color=['#2E86AB', '#A23B72'][:len(methods)], 
                      edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Average Time (seconds)', fontweight='bold')
        ax2.set_title('Computational Cost')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Success Rate
    ax3 = axes[2]
    methods_success = []
    success_rates = []
    
    if gd_all is not None and len(gd_all) > 0:
        methods_success.append('Gradient\nDescent')
        success_rate = (gd_all['success'].sum() / len(gd_all)) * 100
        success_rates.append(success_rate)
    
    if bo_all is not None and len(bo_all) > 0:
        methods_success.append('Bayesian\nOptimization')
        success_rate = (bo_all['success'].sum() / len(bo_all)) * 100
        success_rates.append(success_rate)
    
    if len(methods_success) > 0:
        bars = ax3.bar(methods_success, success_rates, 
                      color=['#2E86AB', '#A23B72'][:len(methods_success)], 
                      edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Success Rate (%)', fontweight='bold')
        ax3.set_title('Convergence Reliability')
        ax3.set_ylim([0, 105])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: images/performance_comparison.png")
    plt.close()

def plot_capacity_fade():
    """Plot capacity fade from synthetic data"""
    try:
        summary = pd.read_csv('synthetic_battery_data/summary_statistics.csv')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Battery Performance Degradation', fontsize=16, fontweight='bold')
        
        # Plot 1: Capacity Fade
        ax1 = axes[0]
        ax1.plot(summary['Cycle_Number'], summary['Capacity_mAh'], 'o-', 
                color='#F18F01', linewidth=2, markersize=10, markeredgecolor='black')
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Capacity (mAh)', fontweight='bold')
        ax1.set_title('Capacity Fade Over Cycling')
        ax1.grid(True, alpha=0.3)
        
        # Add percentage annotations
        for _, row in summary.iterrows():
            if row['Cycle_Number'] in [3, 309]:
                ax1.annotate(f"{row['Capacity_mAh']:.0f} mAh\n({row['Capacity_Fade_%']:.1f}%)",
                           xy=(row['Cycle_Number'], row['Capacity_mAh']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='yellow', alpha=0.7))
        
        # Plot 2: Energy Fade
        ax2 = axes[1]
        ax2.plot(summary['Cycle_Number'], abs(summary['Energy_Wh']), 'o-', 
                color='#C73E1D', linewidth=2, markersize=10, markeredgecolor='black')
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.set_ylabel('Energy (Wh)', fontweight='bold')
        ax2.set_title('Energy Fade Over Cycling')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/capacity_energy_fade.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: images/capacity_energy_fade.png")
        plt.close()
        
    except FileNotFoundError:
        print("[WARNING] Summary statistics file not found")

def calculate_error_metrics(gd_metrics, bo_metrics):
    """Calculate prediction error metrics"""
    
    # True values from synthetic data generation
    true_values = {
        3: {'LAM_cathode': 0, 'LAM_anode': 0, 'LLI': 0},
        54: {'LAM_cathode': 1.0, 'LAM_anode': 0.42, 'LLI': 0.10},
        105: {'LAM_cathode': 2.0, 'LAM_anode': 0.83, 'LLI': 0.20},
        156: {'LAM_cathode': 3.0, 'LAM_anode': 1.25, 'LLI': 0.30},
        207: {'LAM_cathode': 4.0, 'LAM_anode': 1.67, 'LLI': 0.40},
        258: {'LAM_cathode': 5.0, 'LAM_anode': 2.08, 'LLI': 0.50},
        309: {'LAM_cathode': 6.0, 'LAM_anode': 2.50, 'LLI': 0.60}
    }
    
    results = {'GD': {}, 'BO': {}}
    
    for method_name, metrics in [('GD', gd_metrics), ('BO', bo_metrics)]:
        if metrics is None or len(metrics) == 0:
            continue
        
        errors_lam_cathode = []
        errors_lam_anode = []
        errors_lli = []
        
        for _, row in metrics.iterrows():
            cycle = int(row['Cycle'])
            if cycle in true_values:
                true = true_values[cycle]
                
                # Calculate errors
                err_lam_c = abs(row['LAM_Cathode_%'] - true['LAM_cathode'])
                err_lam_a = abs(row['LAM_Anode_%'] - true['LAM_anode'])
                err_lli = abs(row['LLI_mAh'] - true['LLI'])
                
                errors_lam_cathode.append(err_lam_c)
                errors_lam_anode.append(err_lam_a)
                errors_lli.append(err_lli)
        
        # Calculate metrics
        if len(errors_lam_cathode) > 0:
            results[method_name] = {
                'MAE_LAM_Cathode': np.mean(errors_lam_cathode),
                'MAE_LAM_Anode': np.mean(errors_lam_anode),
                'MAE_LLI': np.mean(errors_lli),
                'RMSE_LAM_Cathode': np.sqrt(np.mean(np.array(errors_lam_cathode)**2)),
                'RMSE_LAM_Anode': np.sqrt(np.mean(np.array(errors_lam_anode)**2)),
                'RMSE_LLI': np.sqrt(np.mean(np.array(errors_lli)**2))
            }
    
    return results

def generate_metrics_report():
    """Generate comprehensive metrics report"""
    print("\n" + "="*70)
    print("PERFORMANCE METRICS AND ANALYSIS")
    print("="*70 + "\n")
    
    # Load results
    gd_best, gd_all, bo_best, bo_all = load_results()
    
    # Calculate aging metrics
    gd_metrics = calculate_aging_metrics(gd_best, 'Gradient Descent')
    bo_metrics = calculate_aging_metrics(bo_best, 'Bayesian Optimization')
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_capacity_fade()
    plot_aging_comparison(gd_metrics, bo_metrics)
    plot_performance_comparison(gd_all, bo_all)
    
    # Calculate error metrics
    print("\nCalculating prediction errors...")
    error_metrics = calculate_error_metrics(gd_metrics, bo_metrics)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for method in ['GD', 'BO']:
        if method in error_metrics and error_metrics[method]:
            print(f"\n{method} - Error Metrics:")
            print(f"  MAE LAM Cathode:  {error_metrics[method]['MAE_LAM_Cathode']:.3f}%")
            print(f"  MAE LAM Anode:    {error_metrics[method]['MAE_LAM_Anode']:.3f}%")
            print(f"  MAE LLI:          {error_metrics[method]['MAE_LLI']:.3f} mAh")
            print(f"  RMSE LAM Cathode: {error_metrics[method]['RMSE_LAM_Cathode']:.3f}%")
            print(f"  RMSE LAM Anode:   {error_metrics[method]['RMSE_LAM_Anode']:.3f}%")
            print(f"  RMSE LLI:         {error_metrics[method]['RMSE_LLI']:.3f} mAh")
    
    # Save metrics to JSON
    with open('results/error_metrics.json', 'w') as f:
        json.dump(error_metrics, f, indent=2)
    print("\n[OK] Saved: results/error_metrics.json")
    
    # Save combined metrics
    if gd_metrics is not None and bo_metrics is not None:
        combined_metrics = pd.concat([gd_metrics, bo_metrics], ignore_index=True)
        combined_metrics.to_csv('results/combined_aging_metrics.csv', index=False)
        print("[OK] Saved: results/combined_aging_metrics.csv")
    
    print("\n" + "="*70)
    print("METRICS ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - images/capacity_energy_fade.png")
    print("  - images/aging_comparison.png")
    print("  - images/performance_comparison.png")
    print("  - results/error_metrics.json")
    print("  - results/combined_aging_metrics.csv")
    print("\n")

if __name__ == "__main__":
    generate_metrics_report()
