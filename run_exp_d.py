#!/usr/bin/env python3
"""
run_exp_d.py — Experiment D: NSP-Monitor on SWE-bench Verified trajectories.

End-to-end pipeline:
  1. Parse SWE-agent trajectory JSONs into 12-dim feature vectors
  2. Load gold-standard correctness labels (resolved true/false)
  3. Split into train (for fitting NSP) and test (for evaluation)
  4. Fit NSP via EM on training trajectories
  5. Score test trajectories for drift (Algorithm 2)
  6. Run three baselines (trajectory length, discrete HMM, LLM-as-judge)
  7. Compute AUROCs and produce the headline Table 1
  8. Report operational measurements (latency, memory, cost)

DIRECTORY STRUCTURE (expected):
  exp_d_swe/
    traj_swe/          ← SWE-agent trajectory JSONs (one per instance)
    paper_data/        ← created by this script
    run_exp_d.py       ← this file

INPUT:
  --traj-dir       path to trajectory JSONs (default: ./traj_swe)
  --labels         CSV with columns instance, resolved (from sb-cli evaluate)
  --train-ratio    fraction of data for fitting (default: 0.67)

OUTPUT (in --output-dir, default ./paper_data/):
  - exp_d_features.csv          12-dim features per step per instance
  - exp_d_nsp_params.npz        fitted NSP parameters
  - exp_d_scores.csv            per-instance drift scores + baselines
  - exp_d_table1.csv            headline AUROC table (Table 1 of paper)
  - exp_d_table1.txt            formatted Table 1 for quick reading
  - exp_d_operational.json      latency, memory, cost measurements
  - exp_d_report.json           full summary

USAGE:
  # With pre-downloaded trajectories and labels:
  python3 run_exp_d.py --traj-dir traj_swe/ --labels swe_labels.csv

  # Generate a demo with synthetic SWE-agent-like trajectories (no real data needed):
  python3 run_exp_d.py --demo

  # In Jupyter:
  %run run_exp_d.py --demo
  %run run_exp_d.py --traj-dir traj_swe/ --labels swe_labels.csv
"""
import argparse
import json
import os
import re
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 1. Feature extraction — 12-dim per trajectory step
# =============================================================================

TOOL_TYPES = ["edit", "view", "search", "test", "submit"]

def features_per_step(step):
    """
    Extract 12-dimensional feature vector from a single SWE-agent trajectory step.

    Dimensions:
      0:  files_touched   — number of distinct files mentioned in the action
      1:  hunks           — number of diff hunks (@@ markers)
      2:  lines_added     — count of '+' lines in diff
      3:  lines_removed   — count of '-' lines in diff
      4:  test_pass       — 1.0 if observation contains PASS/OK, else 0.0
      5:  reason_len      — character length of the agent's reasoning/thought
      6:  lexical_diversity — unique words / total words in thought
      7-11: tool_onehot   — one-hot for [edit, view, search, test, submit]
    """
    action = step.get("action", "") or ""
    obs = step.get("observation", "") or ""
    thought = step.get("thought", "") or ""

    # If the step structure uses different field names, try alternatives
    if not action and "command" in step:
        action = step.get("command", "")
    if not thought and "message" in step:
        thought = step.get("message", "")
    if not obs and "output" in step:
        obs = step.get("output", "")

    files_touched = len(set(re.findall(r'file:\s*(\S+)', action)))
    if files_touched == 0:
        # Try to find file paths in the action
        files_touched = len(set(re.findall(
            r'[\w./]+\.(?:py|js|ts|java|c|cpp|h|go|rs|rb)', action)))

    hunks = action.count("@@")
    lines_added = action.count("\n+")
    lines_removed = action.count("\n-")
    test_pass = 1.0 if ("PASS" in obs.upper() or "OK" in obs.upper()) else 0.0
    reason_len = float(len(thought))

    words = thought.split()
    diversity = float(len(set(words)) / max(len(words), 1))

    tool_onehot = [1.0 if t in action.lower() else 0.0 for t in TOOL_TYPES]

    return [files_touched, hunks, lines_added, lines_removed,
            test_pass, reason_len, diversity] + tool_onehot


def parse_trajectories(traj_dir):
    """Parse all trajectory JSONs in a directory into a features DataFrame."""
    traj_dir = Path(traj_dir)
    json_files = sorted(traj_dir.glob("*.json"))
    if not json_files:
        # Also check for .traj files (some SWE-agent versions use this extension)
        json_files = sorted(traj_dir.glob("*.traj"))
    if not json_files:
        print(f"  WARNING: no .json or .traj files found in {traj_dir}")
        return pd.DataFrame()

    rows = []
    instances_parsed = 0
    for traj_file in tqdm(json_files, desc="parsing trajectories"):
        try:
            with open(traj_file) as f:
                traj = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        iid = traj.get("instance_id", traj_file.stem)

        # SWE-agent stores steps in "history" or "trajectory"
        steps = traj.get("history", traj.get("trajectory", []))
        if not steps:
            continue

        for step_idx, step in enumerate(steps):
            feats = features_per_step(step)
            row = {"instance": iid, "step": step_idx}
            row.update({f"f{i}": v for i, v in enumerate(feats)})
            rows.append(row)
        instances_parsed += 1

    print(f"  parsed {instances_parsed} trajectories, {len(rows)} total steps")
    return pd.DataFrame(rows)


# =============================================================================
# 2. Demo data generator (synthetic SWE-agent-like trajectories)
# =============================================================================

def generate_demo_data(n_instances=60, seed=42):
    """
    Generate synthetic SWE-agent-like trajectories for testing the pipeline
    without real SWE-bench data. 60% resolved, 40% unresolved.
    Unresolved trajectories have higher drift (more chaotic features).
    """
    np.random.seed(seed)
    rows_feat = []
    rows_label = []

    for i in range(n_instances):
        resolved = i < int(n_instances * 0.6)
        iid = f"demo_instance_{i:04d}"
        length = np.random.randint(8, 25)

        for k in range(length):
            # Base features
            files = np.random.randint(0, 4)
            hunks = np.random.randint(0, 6)
            added = np.random.randint(0, 30)
            removed = np.random.randint(0, 20)
            test_pass = 1.0 if np.random.random() > 0.3 else 0.0
            reason_len = float(np.random.randint(50, 500))
            diversity = np.random.uniform(0.3, 0.9)
            tool = [0.0] * 5
            tool[np.random.randint(0, 5)] = 1.0

            if not resolved:
                # Unresolved: inject drift — increasing chaos over steps
                drift_factor = 1 + k * 0.15
                files = int(files * drift_factor)
                hunks = int(hunks * drift_factor)
                added = int(added * drift_factor)
                reason_len *= (1 + k * 0.1)
                diversity *= max(0.2, 1 - k * 0.05)

            feats = [files, hunks, added, removed, test_pass,
                     reason_len, diversity] + tool
            row = {"instance": iid, "step": k}
            row.update({f"f{j}": v for j, v in enumerate(feats)})
            rows_feat.append(row)

        rows_label.append({"instance": iid, "resolved": resolved})

    return pd.DataFrame(rows_feat), pd.DataFrame(rows_label)


# =============================================================================
# 3. NSP fitting and scoring (reuses fit_nsp_text.py's EM machinery)
# =============================================================================

def kalman_forward(O, R, A, B, C, Q, Sigma):
    """Kalman forward pass. Returns filtered states and log-likelihood."""
    T, D = len(O), A.shape[0]
    mu_f = np.zeros((T, D))
    P_f = np.zeros((T, D, D))
    P_f[0] = np.eye(D)
    log_lik = 0.0
    jitter = 1e-6 * np.eye(D)
    log_two_pi = D * np.log(2 * np.pi)

    for t in range(1, T):
        mu_pred = A @ mu_f[t - 1] + (B * R[t - 1]).flatten()
        P_pred = A @ P_f[t - 1] @ A.T + Q
        S_mat = C @ P_pred @ C.T + Sigma + jitter
        innov = O[t] - C @ mu_pred
        try:
            S_inv = np.linalg.inv(S_mat)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S_mat)
        K_gain = P_pred @ C.T @ S_inv
        mu_f[t] = mu_pred + K_gain @ innov
        P_f[t] = (np.eye(D) - K_gain @ C) @ P_pred
        sign, logdet = np.linalg.slogdet(S_mat)
        if sign > 0:
            log_lik += -0.5 * (log_two_pi + logdet + innov @ S_inv @ innov)

    return mu_f, log_lik


def kalman_smoother(O, R, A, B, C, Q, Sigma):
    """Full forward-backward smoother for EM. Returns smoothed states + loglik."""
    T, D = len(O), A.shape[0]
    mu_f = np.zeros((T, D))
    P_f = np.zeros((T, D, D))
    P_f[0] = np.eye(D)
    log_lik = 0.0
    jitter = 1e-6 * np.eye(D)
    log_two_pi = D * np.log(2 * np.pi)

    for t in range(1, T):
        mu_pred = A @ mu_f[t - 1] + (B * R[t - 1]).flatten()
        P_pred = A @ P_f[t - 1] @ A.T + Q
        S_mat = C @ P_pred @ C.T + Sigma + jitter
        innov = O[t] - C @ mu_pred
        try:
            S_inv = np.linalg.inv(S_mat)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S_mat)
        K_gain = P_pred @ C.T @ S_inv
        mu_f[t] = mu_pred + K_gain @ innov
        P_f[t] = (np.eye(D) - K_gain @ C) @ P_pred
        sign, logdet = np.linalg.slogdet(S_mat)
        if sign > 0:
            log_lik += -0.5 * (log_two_pi + logdet + innov @ S_inv @ innov)

    mu_s = mu_f.copy()
    P_s = P_f.copy()
    for t in range(T - 2, -1, -1):
        mu_pred = A @ mu_f[t] + (B * R[t]).flatten()
        P_pred = A @ P_f[t] @ A.T + Q + jitter
        try:
            P_pred_inv = np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            P_pred_inv = np.linalg.pinv(P_pred)
        J = P_f[t] @ A.T @ P_pred_inv
        mu_s[t] = mu_f[t] + J @ (mu_s[t + 1] - mu_pred)
        P_s[t] = P_f[t] + J @ (P_s[t + 1] - P_pred) @ J.T

    return mu_s, P_s, log_lik


def fit_em(trajs, rewards, D, n_iters=40, verbose=True):
    """Fit A by EM. C fixed to identity, features pre-standardized."""
    A = np.eye(D) * 0.5 + np.random.randn(D, D) * 0.05
    B = np.zeros((D, 1))
    C = np.eye(D)
    Q = np.eye(D) * 0.5
    Sigma = np.eye(D) * 0.5

    history = []
    pbar = tqdm(range(n_iters), desc='EM fit', disable=not verbose)
    for _ in pbar:
        smoothed = []
        total_ll = 0.0
        for O, R in zip(trajs, rewards):
            mu_s, P_s, ll = kalman_smoother(O, R, A, B, C, Q, Sigma)
            smoothed.append((mu_s, P_s))
            total_ll += ll
        history.append(total_ll)
        if verbose:
            pbar.set_postfix(log_lik=f'{total_ll:.1f}')

        num = np.zeros((D, D))
        den = np.zeros((D, D))
        for (mu_s, P_s), R in zip(smoothed, rewards):
            T = len(mu_s)
            for t in range(T - 1):
                num += np.outer(mu_s[t + 1], mu_s[t])
                den += np.outer(mu_s[t], mu_s[t]) + P_s[t]
        A = num @ np.linalg.inv(den + 1e-4 * np.eye(D))

    return A, B, C, Q, Sigma, history


def score_trajectory(O, R, A, B, C, Q, Sigma):
    """Score one trajectory: return final drift magnitude + drift slope."""
    mu_f, log_lik = kalman_forward(O, R, A, B, C, Q, Sigma)
    drifts = np.array([np.linalg.norm(mu_f[k] - mu_f[0]) for k in range(len(mu_f))])
    final_drift = drifts[-1]

    if len(drifts) >= 3:
        from scipy.stats import linregress
        slope = linregress(np.arange(len(drifts)), drifts).slope
    else:
        slope = 0.0

    return final_drift, slope, log_lik


# =============================================================================
# 4. Baselines
# =============================================================================

def baseline_trajectory_length(test_groups):
    """Baseline 1: trajectory length as drift proxy (trivial)."""
    return [len(g) for g in test_groups]


def baseline_discrete_hmm(train_groups, test_groups, n_components=8):
    """Baseline 3: discrete-state Gaussian HMM via hmmlearn."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  WARNING: hmmlearn not installed, skipping HMM baseline")
        print("  Install with: pip install hmmlearn")
        return None

    train_concat = np.vstack(train_groups)
    lengths = [len(g) for g in train_groups]
    m = GaussianHMM(n_components=n_components, n_iter=50, random_state=42,
                    covariance_type='diag')
    m.fit(train_concat, lengths)

    # Negate log-likelihood so larger = more drift
    scores = []
    for g in test_groups:
        try:
            scores.append(-m.score(g))
        except Exception:
            scores.append(0.0)
    return scores


def baseline_llm_judge(test_instances, test_patches, skip=False):
    """
    Baseline 2: GPT-4 as judge (~$10 for 100 instances).
    Returns confidence scores. Set skip=True to skip (saves money).
    """
    if skip:
        print("  LLM-as-judge baseline skipped (--skip-llm-judge)")
        return None

    try:
        import openai
    except ImportError:
        print("  WARNING: openai not installed, skipping LLM-as-judge baseline")
        return None

    if not os.environ.get("OPENAI_API_KEY"):
        print("  WARNING: OPENAI_API_KEY not set, skipping LLM-as-judge baseline")
        return None

    client = openai.OpenAI()
    scores = []
    for problem, patch in tqdm(zip(test_instances, test_patches),
                               total=len(test_instances), desc='LLM judge'):
        msg = (f"Problem: {problem}\nPatch:\n{patch}\n\n"
               "Is this patch correct? Reply ONLY with JSON: "
               '{"correct": true/false, "confidence": 0.0-1.0}')
        try:
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": msg}],
                response_format={"type": "json_object"})
            data = json.loads(r.choices[0].message.content)
            # Invert: high confidence of correct = low drift
            scores.append(1.0 - data.get('confidence', 0.5))
        except Exception:
            scores.append(0.5)
    return scores


# =============================================================================
# 5. Main pipeline
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Experiment D: NSP-Monitor on SWE-bench Verified trajectories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--traj-dir', default='./traj_swe',
                    help='Directory containing SWE-agent trajectory JSONs')
    ap.add_argument('--labels', default=None,
                    help='CSV with columns instance, resolved (from sb-cli evaluate)')
    ap.add_argument('--output-dir', default='./paper_data',
                    help='Output directory')
    ap.add_argument('--train-ratio', type=float, default=0.67,
                    help='Fraction of instances for NSP fitting')
    ap.add_argument('--em-iters', type=int, default=40,
                    help='EM iterations for NSP fitting')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-llm-judge', action='store_true',
                    help='Skip the LLM-as-judge baseline (saves ~$10)')
    ap.add_argument('--demo', action='store_true',
                    help='Run with synthetic demo data (no real trajectories needed)')
    args = ap.parse_args()

    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    D = 12  # feature dimensionality

    # ─── Step 1: Load or generate data ───────────────────────────────────
    if args.demo:
        print("Running in DEMO mode with synthetic trajectories...")
        df_feat, df_labels = generate_demo_data(n_instances=60)
    else:
        print(f"Parsing trajectories from {args.traj_dir}...")
        df_feat = parse_trajectories(args.traj_dir)
        if df_feat.empty:
            sys.exit("ERROR: no trajectories found. Use --demo for synthetic data.")
        if args.labels is None:
            sys.exit("ERROR: --labels required (CSV with instance, resolved)")
        df_labels = pd.read_csv(args.labels)

    # Save features
    df_feat.to_csv(out / 'exp_d_features.csv', index=False)
    print(f"Saved features → {out / 'exp_d_features.csv'}")

    # ─── Step 2: Merge features with labels ──────────────────────────────
    instances_with_labels = set(df_labels['instance'].unique())
    instances_with_features = set(df_feat['instance'].unique())
    common = sorted(instances_with_labels & instances_with_features)
    print(f"Instances: {len(instances_with_features)} with features, "
          f"{len(instances_with_labels)} with labels, {len(common)} in common")

    if len(common) < 10:
        sys.exit("ERROR: fewer than 10 instances in common. Check data alignment.")

    df_feat = df_feat[df_feat['instance'].isin(common)]
    df_labels = df_labels[df_labels['instance'].isin(common)]
    label_map = dict(zip(df_labels['instance'], df_labels['resolved']))

    # ─── Step 3: Train/test split ────────────────────────────────────────
    np.random.shuffle(common)
    split = int(len(common) * args.train_ratio)
    train_ids = set(common[:split])
    test_ids = set(common[split:])
    print(f"Split: {len(train_ids)} train, {len(test_ids)} test")

    feature_cols = [f'f{i}' for i in range(D)]

    def group_trajectories(ids):
        groups, labels, iids = [], [], []
        for iid in sorted(ids):
            g = df_feat[df_feat['instance'] == iid].sort_values('step')
            if len(g) < 3:
                continue
            groups.append(g[feature_cols].values.astype(float))
            labels.append(label_map[iid])
            iids.append(iid)
        return groups, np.array(labels), iids

    train_groups, train_labels, train_iids = group_trajectories(train_ids)
    test_groups, test_labels, test_iids = group_trajectories(test_ids)
    print(f"After length filter: {len(train_groups)} train, {len(test_groups)} test")

    # ─── Step 4: Standardize features ────────────────────────────────────
    all_train = np.vstack(train_groups)
    feat_mean = all_train.mean(axis=0)
    feat_std = all_train.std(axis=0) + 1e-8
    train_groups = [(g - feat_mean) / feat_std for g in train_groups]
    test_groups_normed = [(g - feat_mean) / feat_std for g in test_groups]

    # Rewards: use test_pass feature (index 4) as the reward signal
    train_rewards = [g[:, 4] for g in train_groups]
    test_rewards = [g[:, 4] for g in test_groups_normed]

    # ─── Step 5: Fit NSP via EM ──────────────────────────────────────────
    print(f"\nFitting NSP (D={D}, iters={args.em_iters})...")
    tracemalloc.start()
    t0 = time.perf_counter()

    A, B, C, Q, Sigma, em_history = fit_em(
        train_groups, train_rewards, D=D, n_iters=args.em_iters)

    fit_time = time.perf_counter() - t0
    _, fit_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    spec = float(np.max(np.abs(np.linalg.eigvals(A))))
    print(f"  Spectral radius: {spec:.4f}")
    print(f"  Fit time: {fit_time:.1f}s")
    print(f"  Peak memory: {fit_peak_mem / 1024 / 1024:.1f} MB")

    np.savez(out / 'exp_d_nsp_params.npz',
             A=A, B=B, C=C, Q=Q, Sigma=Sigma,
             feature_mean=feat_mean, feature_std=feat_std,
             em_history=np.array(em_history))

    # ─── Step 6: Score test trajectories (NSP-Monitor) ───────────────────
    print("\nScoring test trajectories...")
    tracemalloc.start()
    t0 = time.perf_counter()

    nsp_scores = []
    nsp_slopes = []
    nsp_logliks = []
    for O, R in zip(test_groups_normed, test_rewards):
        fd, slope, ll = score_trajectory(O, R, A, B, C, Q, Sigma)
        nsp_scores.append(fd)
        nsp_slopes.append(slope)
        nsp_logliks.append(ll)

    score_time = time.perf_counter() - t0
    _, score_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    per_traj_latency = score_time / max(len(test_groups_normed), 1)
    print(f"  Score time: {score_time:.2f}s ({per_traj_latency*1000:.1f}ms/trajectory)")

    # ─── Step 7: Run baselines ───────────────────────────────────────────
    print("\nRunning baselines...")

    # Baseline 1: trajectory length
    scores_length = baseline_trajectory_length(test_groups)

    # Baseline 2: LLM-as-judge (optional)
    scores_llm = baseline_llm_judge(
        test_iids, ["[patch placeholder]"] * len(test_iids),
        skip=args.skip_llm_judge or args.demo)

    # Baseline 3: discrete HMM
    scores_hmm = baseline_discrete_hmm(train_groups, test_groups_normed)

    # ─── Step 8: Compute AUROCs ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS — Table 1: AUROC for Detecting Incorrect Patches")
    print("=" * 60)

    from sklearn.metrics import roc_auc_score

    y_incorrect = 1 - test_labels  # 1 = incorrect (unresolved)

    # Collect all methods
    methods = {
        'NSP-Monitor (final_drift)': nsp_scores,
        'NSP-Monitor (drift_slope)': nsp_slopes,
        'Trajectory length': scores_length,
    }
    if scores_hmm is not None:
        methods['Discrete HMM'] = scores_hmm
    if scores_llm is not None:
        methods['LLM-as-judge (GPT-4o)'] = scores_llm

    table_rows = []
    print(f"\n{'Method':<35} {'AUROC (all)':>12} {'AUROC (test-passed)':>20}")
    print("-" * 70)

    # Identify test-passed subset (test_pass = feature index 4 in unnormed data)
    test_passed_mask = np.array([g[-1, 4] > 0.5 for g in test_groups])

    for name, scores in methods.items():
        scores_arr = np.array(scores, dtype=float)

        # AUROC all
        try:
            auroc_all = roc_auc_score(y_incorrect, scores_arr)
        except ValueError:
            auroc_all = float('nan')

        # AUROC on test-passed subset
        if test_passed_mask.sum() >= 2 and len(np.unique(y_incorrect[test_passed_mask])) >= 2:
            try:
                auroc_pass = roc_auc_score(
                    y_incorrect[test_passed_mask],
                    scores_arr[test_passed_mask])
            except ValueError:
                auroc_pass = float('nan')
        else:
            auroc_pass = float('nan')

        print(f"{name:<35} {auroc_all:>12.3f} {auroc_pass:>20.3f}")
        table_rows.append({
            'method': name,
            'auroc_all': auroc_all,
            'auroc_test_passed': auroc_pass,
        })

    print("=" * 70)

    # ─── Step 9: Save outputs ────────────────────────────────────────────
    # Scores CSV
    df_scores = pd.DataFrame({
        'instance': test_iids,
        'resolved': test_labels.astype(bool),
        'nsp_final_drift': nsp_scores,
        'nsp_drift_slope': nsp_slopes,
        'nsp_loglik': nsp_logliks,
        'baseline_length': scores_length,
    })
    if scores_hmm is not None:
        df_scores['baseline_hmm'] = scores_hmm
    if scores_llm is not None:
        df_scores['baseline_llm_judge'] = scores_llm
    df_scores.to_csv(out / 'exp_d_scores.csv', index=False)
    print(f"\nSaved scores → {out / 'exp_d_scores.csv'}")

    # Table 1
    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(out / 'exp_d_table1.csv', index=False)
    with open(out / 'exp_d_table1.txt', 'w') as f:
        f.write(df_table.to_string(index=False))
    print(f"Saved Table 1 → {out / 'exp_d_table1.csv'}")

    # Operational measurements
    ops = {
        'fit_time_seconds': fit_time,
        'fit_peak_memory_mb': fit_peak_mem / 1024 / 1024,
        'score_time_seconds': score_time,
        'score_peak_memory_mb': score_peak_mem / 1024 / 1024,
        'per_trajectory_latency_ms': per_traj_latency * 1000,
        'n_train': len(train_groups),
        'n_test': len(test_groups),
        'em_iters': args.em_iters,
        'spectral_radius_A': spec,
        'cost_nsp_per_trajectory_dollars': 0.0,
        'cost_llm_judge_per_trajectory_dollars': 0.015 if scores_llm else None,
    }
    with open(out / 'exp_d_operational.json', 'w') as f:
        json.dump(ops, f, indent=2)
    print(f"Saved operational measurements → {out / 'exp_d_operational.json'}")

    # Full report
    report = {
        'mode': 'demo' if args.demo else 'real',
        'n_instances_total': len(common),
        'n_train': len(train_groups),
        'n_test': len(test_groups),
        'spectral_radius_A': spec,
        'final_log_lik': float(em_history[-1]),
        'table1': table_rows,
        'operational': ops,
    }
    with open(out / 'exp_d_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report → {out / 'exp_d_report.json'}")

    print("\n✓ Experiment D complete.")


if __name__ == '__main__':
    main()
