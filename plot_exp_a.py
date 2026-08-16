"""
Plot Experiment A results: MSE vs N on a log-log scale.
Reproduces Figure 2 of the SEIP paper (Figure 3 of the NSP paper).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
df = pd.read_csv('../paper_data/exp_a_results.csv')

# Split into two conditions
regular = df[df['rank_deficient'] == False]
rank_def = df[df['rank_deficient'] == True]

# Cramér-Rao reference line (theoretical best possible rate)
# This is a straight line on log-log scale with slope -1
N_range = np.array([100, 500, 1000])
crb = 0.5 / N_range  # scales as 1/N

# Create the plot
fig, ax = plt.subplots(figsize=(7, 5))

# Plot regular condition
ax.errorbar(regular['N'], regular['mse_mean'],
            yerr=regular['mse_std'],
            marker='o', linewidth=2, markersize=6,
            label='Regular (R1–R4 hold)',
            color='steelblue', capsize=4)

# Plot rank-deficient condition
ax.errorbar(rank_def['N'], rank_def['mse_mean'],
            yerr=rank_def['mse_std'],
            marker='s', linewidth=2, markersize=6,
            label='Rank-deficient (R2 violated)',
            color='tomato', capsize=4)

# Plot Cramér-Rao bound reference line
ax.plot(N_range, crb,
        linestyle='--', linewidth=1.5,
        label='CRB reference (slope −1)',
        color='gray')

# Log-log scale
ax.set_xscale('log')
ax.set_yscale('log')

# Labels and formatting
ax.set_xlabel('Number of trajectories (N)', fontsize=12)
ax.set_ylabel('Parameter MSE', fontsize=12)
ax.set_title('Synthetic Recovery under Linear-Gaussian NSP', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.4)

plt.tight_layout()

# Save the figure
out_path = Path('../paper_data/exp_a_loglog.png')
plt.savefig(out_path, dpi=150)
print(f'Plot saved to {out_path}')
plt.show()