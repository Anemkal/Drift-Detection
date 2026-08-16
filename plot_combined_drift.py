"""
Combined drift plot showing NSP drift scores across all image tests.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
TESTS = [
    {
        'name': 'Test_3',
        'json': 'images/Test_3/results_json/results_prompt_1b9bf0d31a76.json',
        'label': 'Test 3 (Google, 15 iter)',
        'color': 'steelblue',
        'marker': 'o',
    },
    {
        'name': 'Test_4',
        'json': 'images/Test_4/results_json/results_prompt_0a39865ff5a2.json',
        'label': 'Test 4 (Google, 15 iter)',
        'color': 'tomato',
        'marker': 's',
    },
    {
        'name': 'Test_5',
        'json': 'images/Test_5/results_json/results_prompt_d48e0c7d1485.json',
        'label': 'Test 5 (OpenAI, 15 iter, people)',
        'color': 'green',
        'marker': '^',
    },
    {
        'name': 'Test_7',
        'json': 'images/Test_7/results_json/results_prompt_2b0f508e1ebd.json',
        'label': 'Test 7 (OpenAI, 8 iter, robots)',
        'color': 'purple',
        'marker': 'D',
    },
    {
        'name': 'Test_8',
        'json': 'images/Test_8/results_json/results_prompt_a147908e4fb9.json',
        'label': 'Test 8 (OpenAI, 5 iter, robots+dinos)',
        'color': 'orange',
        'marker': 'P',
    },
]

METRIC_KEYS = [
    'emotional_valence',
    'sentiment_polarity',
    'arousal_level',
    'dominance',
    'facial_expression_score',
    'color_warmth_index',
    'compositional_tension',
]

def compute_drift(json_path):
    """Load JSON and compute NSP drift scores."""
    with open(json_path) as f:
        data = json.load(f)

    # Build observations list starting with baseline
    observations = []
    baseline_metrics = data['baseline_metrics']['metrics']
    observations.append([baseline_metrics[k] for k in METRIC_KEYS])

    for it in data['iterations']:
        observations.append([it['metrics'][k] for k in METRIC_KEYS])

    observations = np.array(observations)

    # Standardize
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0) + 1e-8
    obs_norm = (observations - obs_mean) / obs_std

    # Compute drift as L2 distance from baseline
    baseline = obs_norm[0]
    drift_scores = [float(np.linalg.norm(obs_norm[i] - baseline))
                    for i in range(len(obs_norm))]

    iterations = list(range(len(drift_scores)))
    return iterations, drift_scores


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('NSP Drift Score Comparison Across All Image Tests',
             fontsize=14)

# Left plot: all tests on same axes
ax1 = axes[0]
for test in TESTS:
    iters, scores = compute_drift(test['json'])
    ax1.plot(iters, scores,
             marker=test['marker'],
             color=test['color'],
             linewidth=2,
             label=test['label'])

ax1.set_title('All Tests — NSP Drift Score')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Drift from Baseline')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Right plot: normalized to show drift rate (0 to 1 scale)
ax2 = axes[1]
for test in TESTS:
    iters, scores = compute_drift(test['json'])
    max_score = max(scores) if max(scores) > 0 else 1
    normalized = [s / max_score for s in scores]
    ax2.plot(iters, normalized,
             marker=test['marker'],
             color=test['color'],
             linewidth=2,
             label=test['label'])

ax2.set_title('All Tests — Normalized Drift (0 to 1 scale)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Normalized Drift')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path('paper_data/combined_drift_plot.png')
plt.savefig(out_path, dpi=150)
print(f'Plot saved to {out_path}')
plt.show()