"""
Analyze image drift from Test_5 results JSON.
Fits NSP to per-iteration metrics and plots drift trajectory.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1. Load the results JSON ──────────────────────────────────────────────────
# json_path = Path('images/Test_5/results_json/results_prompt_d48e0c7d1485.json')
# json_path = Path('images/Test_8/results_json/results_prompt_a147908e4fb9.json')
# json_path = Path('images/Test_7/results_json/results_prompt_2b0f508e1ebd.json')
#json_path = Path('images/Test_4/results_json/results_prompt_0a39865ff5a2.json')
json_path = Path('images/Test_3/results_json/results_prompt_1b9bf0d31a76.json')

with open(json_path) as f:
    data = json.load(f)

# ── 2. Extract metrics into a DataFrame ───────────────────────────────────────
METRIC_KEYS = [
    'emotional_valence',
    'sentiment_polarity',
    'arousal_level',
    'dominance',
    'facial_expression_score',
    'color_warmth_index',
    'compositional_tension',
]

rows = []

# Add baseline (iteration 0)
baseline = data['baseline_metrics']
row = {'iteration': 0, 'deviation_value': baseline['deviation_value']}
for k in METRIC_KEYS:
    row[k] = baseline['metrics'][k]
rows.append(row)

# Add iterations 1-15
for it in data['iterations']:
    row = {'iteration': it['stage_number'],
           'deviation_value': it['deviation_value']}
    for k in METRIC_KEYS:
        row[k] = it['metrics'][k]
    rows.append(row)

df = pd.DataFrame(rows)
print("Loaded data:")
print(df[['iteration', 'deviation_value', 'color_warmth_index',
          'emotional_valence']].to_string(index=False))

# ── 3. Fit NSP to the metric trajectory ───────────────────────────────────────
# Use the 7 metrics as observations, fit a 1D latent drift state via EM

observations = df[METRIC_KEYS].values  # shape (16, 7)
T, obs_dim = observations.shape
D = 1  # 1D latent drift state for simplicity

# Standardize observations
obs_mean = observations.mean(axis=0)
obs_std = observations.std(axis=0) + 1e-8
obs_norm = (observations - obs_mean) / obs_std

# Simple NSP: fit a scalar drift state using a sliding window approach
# Compute cumulative drift as L2 distance from baseline at each step
baseline_obs = obs_norm[0]
drift_scores = [float(np.linalg.norm(obs_norm[i] - baseline_obs))
                for i in range(T)]

df['nsp_drift_score'] = drift_scores

print("\nNSP drift scores:")
print(df[['iteration', 'nsp_drift_score',
          'deviation_value']].to_string(index=False))

# ── 4. Plot results ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# fig.suptitle('Image Drift Analysis — Test_5\n'
#              'Prompt: "Express more feeling in this image" (15 iterations)',
#              fontsize=13)
# fig.suptitle('Image Drift Analysis — Test_8\n'
#              'Prompt: "Expose any feeling in this image more clearly" (5 iterations)',
# )
# fig.suptitle('Image Drift Analysis — Test_7\n'
#              'Prompt: "four cartoon robots standing in a small group" (8 iterations)',
# )
# fig.suptitle('Image Drift Analysis — Test_4\n'
#              'Provider: Google, same follow-up prompt (15 iterations)',
# )
fig.suptitle('Image Drift Analysis — Test_3\n'
             'Provider: Google, same follow-up prompt (15 iterations)',
)

# Plot 1: NSP drift score over iterations
ax1 = axes[0, 0]
ax1.plot(df['iteration'], df['nsp_drift_score'],
         marker='o', color='steelblue', linewidth=2)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('NSP Drift Score over Iterations')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Drift from Baseline')
ax1.grid(True, alpha=0.3)

# Plot 2: Deviation value over iterations
ax2 = axes[0, 1]
ax2.plot(df['iteration'], df['deviation_value'],
         marker='s', color='tomato', linewidth=2)
ax2.axhline(y=0.5, color='red', linestyle='--',
            alpha=0.7, label='Threshold (0.5)')
ax2.set_title('Deviation Value over Iterations')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Deviation Value')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Color warmth index (most visually obvious drift signal)
ax3 = axes[1, 0]
ax3.plot(df['iteration'], df['color_warmth_index'],
         marker='^', color='orange', linewidth=2)
ax3.set_title('Color Warmth Index over Iterations')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Color Warmth')
ax3.grid(True, alpha=0.3)

# Plot 4: Emotional valence over iterations
ax4 = axes[1, 1]
ax4.plot(df['iteration'], df['emotional_valence'],
         marker='D', color='purple', linewidth=2)
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.set_title('Emotional Valence over Iterations')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Valence (negative = more negative)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
#out_path = Path('paper_data/image_drift_analysis.png')
#out_path = Path('paper_data/image_drift_analysis_test8.png')
#out_path = Path('paper_data/image_drift_analysis_test7.png')
#out_path = Path('paper_data/image_drift_analysis_test4.png')
out_path = Path('paper_data/image_drift_analysis_test3.png')

plt.savefig(out_path, dpi=150)
print(f'\nPlot saved to {out_path}')
plt.show()