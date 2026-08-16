#!/usr/bin/env python3
"""
run_exp_d_real.py — Experiment D on REAL SWE-agent trajectories.

Pulls real, labeled SWE-agent trajectories + resolved/unresolved outcomes
directly from the SWE-bench/SWE-smith-trajectories HuggingFace dataset,
then runs the same NSP-Monitor pipeline as run_exp_d.py:
  1. Parse trajectories -> 12-dim feature vectors per step
  2. Fit NSP via EM on training trajectories
  3. Score test trajectories for drift
  4. Run baselines (trajectory length, discrete HMM)
  5. Compute AUROCs -> Table 1

USAGE (run this locally, NOT in a sandbox without huggingface.co access):
  pip install datasets numpy pandas tqdm scipy scikit-learn hmmlearn
  python run_exp_d_real.py --n-instances 300 --split tool

OUTPUT (in --output-dir, default ./paper_data/):
  exp_d_features.csv, exp_d_labels.csv, exp_d_nsp_params.npz,
  exp_d_scores.csv, exp_d_table1.csv, exp_d_table1.txt,
  exp_d_operational.json, exp_d_report.json
"""
import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 1. Feature extraction — 12-dim per trajectory step (same as run_exp_d.py)
# =============================================================================

TOOL_TYPES = ["edit", "view", "search", "test", "submit"]

import re

def add_repetition_features(steps):
    """
    Given a trajectory's ordered list of step dicts (each with 'tool_args'),
    compute two additional sequential features per step:
      - file_revisited: 1.0 if this step's file path was already touched
        earlier in this same trajectory, else 0.0
      - is_backtrack: 1.0 if this edit's new_str exactly matches an old_str
        from an earlier edit on the same file (i.e. the agent is undoing a
        change it made previously), else 0.0
    Mutates each step dict in place, adding 'file_revisited' and 'is_backtrack'.
    """
    seen_paths = set()
    # path -> list of (old_str, new_str) seen so far for that path
    edit_history = {}

    for step in steps:
        tool_args = step.get("tool_args") or {}
        path = tool_args.get("path", "")
        old_str = tool_args.get("old_str", "")
        new_str = tool_args.get("new_str", "")
        command = tool_args.get("command", "")

        file_revisited = 1.0 if (path and path in seen_paths) else 0.0

        is_backtrack = 0.0
        if command == "str_replace" and path in edit_history and new_str:
            for prev_old, prev_new in edit_history[path]:
                if new_str == prev_old:
                    is_backtrack = 1.0
                    break

        step["file_revisited"] = file_revisited
        step["is_backtrack"] = is_backtrack

        if path:
            seen_paths.add(path)
            if command == "str_replace":
                edit_history.setdefault(path, []).append((old_str, new_str))

    return steps


def features_per_step(step):
    """Extract 12-dim feature vector from a single trajectory step.
    step is a dict with 'action', 'observation', 'thought' string fields,
    plus 'tool_name' and 'tool_args' (dict) when available from tool_calls."""
    action = step.get("action", "") or ""
    obs = step.get("observation", "") or ""
    thought = step.get("thought", "") or ""
    tool_name = (step.get("tool_name") or "").lower()
    tool_args = step.get("tool_args") or {}

    path = tool_args.get("path", "") or ""
    old_str = tool_args.get("old_str", "") or ""
    new_str = tool_args.get("new_str", "") or ""
    file_text = tool_args.get("file_text", "") or ""
    command_arg = tool_args.get("command", "") or ""

    # files_touched: prefer the structured 'path' arg; fall back to regex on action
    if path:
        files_touched = 1
    else:
        files_touched = len(set(re.findall(r'file:\s*(\S+)', action)))
        if files_touched == 0:
            files_touched = len(set(re.findall(
                r'[\w./]+\.(?:py|js|ts|java|c|cpp|h|go|rs|rb)', action)))

    # hunks/lines: derive from real edit content when this is a str_replace_editor
    # edit-type call, else fall back to counting diff markers in the action string
    if command_arg == "str_replace" and (old_str or new_str):
        hunks = 1
        lines_added = new_str.count("\n") + (1 if new_str else 0)
        lines_removed = old_str.count("\n") + (1 if old_str else 0)
    elif command_arg == "create" and file_text:
        hunks = 1
        lines_added = file_text.count("\n") + 1
        lines_removed = 0
    else:
        hunks = action.count("@@")
        lines_added = action.count("\n+")
        lines_removed = action.count("\n-")

    # test_pass: broaden beyond PASS/OK to common pytest/unittest signals.
    # Real trajectories show patterns like "Ran 4 tests in 0.002s\r\n\r\nOK"
    # (OK preceded by newline, not a space) and "Test passed!"
    obs_upper = obs.upper()
    test_pass = 1.0 if (
        "PASS" in obs_upper or re.search(r'\d+\s+PASSED', obs_upper)
        or re.search(r'(?:^|[\r\n])\s*OK\s*(?:$|[\r\n])', obs_upper)
    ) else 0.0

    reason_len = float(len(thought))
    words = thought.split()
    diversity = float(len(set(words)) / max(len(words), 1))

    # tool_onehot: prefer the actual tool/command name, fall back to action text
    haystack = f"{tool_name} {command_arg} {action}".lower()
    tool_onehot = [1.0 if t in haystack else 0.0 for t in TOOL_TYPES]

    file_revisited = float(step.get("file_revisited", 0.0))
    is_backtrack = float(step.get("is_backtrack", 0.0))

    return ([files_touched, hunks, lines_added, lines_removed,
            test_pass, reason_len, diversity] + tool_onehot
            + [file_revisited, is_backtrack])


# =============================================================================
# 2. Parse SWE-smith-trajectories messages into steps
# =============================================================================

def extract_text(content):
    """content can be a plain string or a list of {'type':'text','text':...} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def messages_to_steps(messages):
    """
    Convert one trajectory's messages list into a list of step dicts
    {'action':..., 'thought':..., 'observation':...} matching the
    features_per_step() schema.

    Assistant messages with message_type == 'action' carry 'action' and
    'thought' directly. The matching 'tool' message (found via tool_call_ids)
    supplies the observation text.
    """
    steps = []
    # Build a lookup from tool_call id -> observation text
    observations_by_call_id = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            obs_text = extract_text(m.get("content"))
            for call_id in m.get("tool_call_ids", []) or []:
                observations_by_call_id[call_id] = obs_text

    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant" and m.get("message_type") == "action":
            action = m.get("action", "") or ""
            thought = m.get("thought", m.get("content", "")) or ""
            if not isinstance(thought, str):
                thought = extract_text(thought)

            # Find matching observation via tool_calls -> id, and parse the
            # real tool name + structured arguments (path/old_str/new_str/etc.)
            obs = ""
            tool_name = ""
            tool_args = {}
            for tc in m.get("tool_calls", []) or []:
                call_id = tc.get("id")
                func = tc.get("function", {}) or {}
                tool_name = func.get("name", "") or tool_name
                raw_args = func.get("arguments", "")
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {}
                elif isinstance(raw_args, dict):
                    tool_args = raw_args
                if call_id in observations_by_call_id:
                    obs = observations_by_call_id[call_id]
                    break

            steps.append({
                "action": action, "thought": thought, "observation": obs,
                "tool_name": tool_name, "tool_args": tool_args,
            })

    return steps


def extract_task_type(instance_id):
    """
    Instance IDs look like:
      django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u
      lincolnloop__python-qrcode.456b01d4.combine_file__svo6amwn
      Project-MONAI__MONAI.a09c1f08.pr_2696

    Returns the task-type bucket (e.g. 'func_pm_ctrl_shuffle', 'combine_file',
    'pr', 'lm_rewrite') used to cluster instances by task difficulty/type.
    """
    parts = instance_id.split(".")
    if len(parts) < 3:
        return "unknown"
    tail = parts[-1]
    tail = re.sub(r'__[a-z0-9]{6,}$', '', tail)
    tail = re.sub(r'^pr_\d+$', 'pr', tail)
    return tail


def parse_hf_dataset(split, n_instances, seed=42, task_types=None):
    """
    Stream N instances from SWE-bench/SWE-smith-trajectories, parse each
    into steps, extract features, and collect resolved labels.

    task_types: optional set of task-type strings (see extract_task_type)
    to restrict to — e.g. {'pr'} or {'lm_rewrite'}. None = no filtering.
    """
    from datasets import load_dataset

    print(f"Streaming '{split}' split of SWE-bench/SWE-smith-trajectories...")
    if task_types:
        print(f"Filtering to task type(s): {sorted(task_types)}")
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    rows = []
    labels = []
    n_seen = 0
    n_kept = 0

    pbar = tqdm(total=n_instances, desc="parsing instances")
    for row in ds:
        n_seen += 1
        if n_kept >= n_instances:
            break

        iid = row["instance_id"]
        resolved = bool(row["resolved"])

        if task_types is not None and extract_task_type(iid) not in task_types:
            continue

        try:
            messages = json.loads(row["messages"])
        except (json.JSONDecodeError, TypeError):
            continue

        steps = messages_to_steps(messages)
        if len(steps) < 3:
            continue  # too short to be useful for drift scoring
        steps = add_repetition_features(steps)

        for step_idx, step in enumerate(steps):
            feats = features_per_step(step)
            r = {"instance": iid, "step": step_idx}
            r.update({f"f{i}": v for i, v in enumerate(feats)})
            rows.append(r)

        labels.append({"instance": iid, "resolved": resolved})
        n_kept += 1
        pbar.update(1)

    pbar.close()
    print(f"Scanned {n_seen} rows, kept {n_kept} instances with >=3 steps, "
          f"{len(rows)} total steps.")

    return pd.DataFrame(rows), pd.DataFrame(labels)


# =============================================================================
# 3. NSP fitting and scoring (identical math to run_exp_d.py)
# =============================================================================

def kalman_smoother(O, R, A, B, C, Q, Sigma):
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


def kalman_forward(O, R, A, B, C, Q, Sigma):
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


def fit_em(trajs, rewards, D, n_iters=40, verbose=True):
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
    mu_f, log_lik = kalman_forward(O, R, A, B, C, Q, Sigma)
    drifts = np.array([np.linalg.norm(mu_f[k] - mu_f[0]) for k in range(len(mu_f))])
    final_drift = drifts[-1]
    if len(drifts) >= 3:
        from scipy.stats import linregress
        slope = linregress(np.arange(len(drifts)), drifts).slope
    else:
        slope = 0.0
    return final_drift, slope, log_lik


def baseline_trajectory_length(test_groups):
    return [len(g) for g in test_groups]


def baseline_discrete_hmm(train_groups, test_groups, n_components=8):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  WARNING: hmmlearn not installed, skipping HMM baseline")
        return None

    train_concat = np.vstack(train_groups)
    lengths = [len(g) for g in train_groups]
    m = GaussianHMM(n_components=n_components, n_iter=50, random_state=42,
                     covariance_type='diag')
    m.fit(train_concat, lengths)

    scores = []
    for g in test_groups:
        try:
            scores.append(-m.score(g))
        except Exception:
            scores.append(0.0)
    return scores


# =============================================================================
# 4. Main pipeline
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Experiment D on real SWE-smith trajectories.")
    ap.add_argument('--split', default='tool', choices=['tool', 'xml', 'ticks'],
                     help='Which SWE-smith-trajectories split to use')
    ap.add_argument('--n-instances', type=int, default=300,
                     help='Number of instances to pull (roadmap wants 300 successful)')
    ap.add_argument('--task-type', default=None,
                     help="Comma-separated task type(s) to restrict to, e.g. 'pr' or "
                          "'lm_rewrite' or 'pr,lm_rewrite'. Default: no filtering (all types pooled).")
    ap.add_argument('--output-dir', default='./paper_data')
    ap.add_argument('--train-ratio', type=float, default=0.67)
    ap.add_argument('--em-iters', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    D = 14  # 12 original + file_revisited + is_backtrack

    # ─── Step 1: Pull real trajectories + labels ─────────────────────────
    task_types = set(args.task_type.split(',')) if args.task_type else None
    df_feat, df_labels = parse_hf_dataset(args.split, args.n_instances, seed=args.seed,
                                            task_types=task_types)
    if df_feat.empty:
        raise SystemExit("ERROR: no usable trajectories parsed. Check dataset access.")

    df_feat.to_csv(out / 'exp_d_features.csv', index=False)
    df_labels.to_csv(out / 'exp_d_labels.csv', index=False)
    print(f"Saved features -> {out / 'exp_d_features.csv'}")
    print(f"Saved labels   -> {out / 'exp_d_labels.csv'}")

    # ─── Step 2: Merge + split ────────────────────────────────────────────
    common = sorted(set(df_labels['instance']) & set(df_feat['instance']))
    print(f"Instances in common: {len(common)}")
    if len(common) < 10:
        raise SystemExit("ERROR: fewer than 10 usable instances. Increase --n-instances.")

    label_map = dict(zip(df_labels['instance'], df_labels['resolved']))
    feature_cols = [f'f{i}' for i in range(D)]

    np.random.shuffle(common)
    split_idx = int(len(common) * args.train_ratio)
    train_ids, test_ids = set(common[:split_idx]), set(common[split_idx:])

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
    print(f"Train: {len(train_groups)}  Test: {len(test_groups)}")

    # ─── Step 3: Standardize ──────────────────────────────────────────────
    all_train = np.vstack(train_groups)
    feat_mean = all_train.mean(axis=0)
    feat_std = all_train.std(axis=0) + 1e-8
    train_groups_n = [(g - feat_mean) / feat_std for g in train_groups]
    test_groups_n = [(g - feat_mean) / feat_std for g in test_groups]
    train_rewards = [g[:, 4] for g in train_groups_n]
    test_rewards = [g[:, 4] for g in test_groups_n]

    # ─── Step 4: Fit NSP ──────────────────────────────────────────────────
    print(f"\nFitting NSP (D={D}, iters={args.em_iters})...")
    tracemalloc.start()
    t0 = time.perf_counter()
    A, B, C, Q, Sigma, em_history = fit_em(train_groups_n, train_rewards, D=D, n_iters=args.em_iters)
    fit_time = time.perf_counter() - t0
    _, fit_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Fit time: {fit_time:.1f}s, spectral radius: "
          f"{np.max(np.abs(np.linalg.eigvals(A))):.4f}")

    np.savez(out / 'exp_d_nsp_params.npz', A=A, B=B, C=C, Q=Q, Sigma=Sigma,
              feature_mean=feat_mean, feature_std=feat_std, em_history=np.array(em_history))

    # ─── Step 5: Score test set ───────────────────────────────────────────
    print("\nScoring test trajectories...")
    tracemalloc.start()
    t0 = time.perf_counter()
    nsp_scores, nsp_slopes, nsp_logliks = [], [], []
    for O, R in zip(test_groups_n, test_rewards):
        fd, slope, ll = score_trajectory(O, R, A, B, C, Q, Sigma)
        nsp_scores.append(fd)
        nsp_slopes.append(slope)
        nsp_logliks.append(ll)
    score_time = time.perf_counter() - t0
    _, score_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    per_traj_latency = score_time / max(len(test_groups_n), 1)

    # ─── Step 6: Baselines ────────────────────────────────────────────────
    print("\nRunning baselines...")
    scores_length = baseline_trajectory_length(test_groups)
    scores_hmm = baseline_discrete_hmm(train_groups_n, test_groups_n)

    # ─── Step 6b: Check whether NSP drift is just re-deriving length ──────
    from scipy.stats import pearsonr, spearmanr
    nsp_scores_arr = np.array(nsp_scores, dtype=float)
    length_arr = np.array(scores_length, dtype=float)

    pear_r, pear_p = pearsonr(nsp_scores_arr, length_arr)
    spear_r, spear_p = spearmanr(nsp_scores_arr, length_arr)
    print(f"\nLength-confound check:")
    print(f"  Pearson  corr(NSP final_drift, trajectory length) = {pear_r:.3f} (p={pear_p:.4f})")
    print(f"  Spearman corr(NSP final_drift, trajectory length) = {spear_r:.3f} (p={spear_p:.4f})")

    # Length-normalized drift: average drift per step, instead of total drift.
    # If raw drift is mostly just tracking length, this should look different.
    nsp_scores_normalized = [
        fd / max(length, 1) for fd, length in zip(nsp_scores, scores_length)
    ]

    # ─── Step 7: AUROC / Table 1 ──────────────────────────────────────────
    from sklearn.metrics import roc_auc_score
    y_incorrect = 1 - test_labels.astype(int)

    methods = {
        'NSP-Monitor (final_drift)': nsp_scores,
        'NSP-Monitor (drift_slope)': nsp_slopes,
        'NSP-Monitor (drift/step, length-normalized)': nsp_scores_normalized,
        'Trajectory length': scores_length,
    }
    if scores_hmm is not None:
        methods['Discrete HMM'] = scores_hmm

    test_passed_mask = np.array([
        (g[-3:, 4] > 0.5).any() if len(g) >= 3 else (g[:, 4] > 0.5).any()
        for g in test_groups
    ])

    print("\n" + "=" * 70)
    print("RESULTS -- Table 1: AUROC for Detecting Incorrect Patches")
    print("=" * 70)
    print(f"\n{'Method':<35} {'AUROC (all)':>12} {'AUROC (test-passed)':>20}")
    print("-" * 70)

    table_rows = []
    for name, scores in methods.items():
        scores_arr = np.array(scores, dtype=float)
        try:
            auroc_all = roc_auc_score(y_incorrect, scores_arr)
        except ValueError:
            auroc_all = float('nan')
        if test_passed_mask.sum() >= 2 and len(np.unique(y_incorrect[test_passed_mask])) >= 2:
            try:
                auroc_pass = roc_auc_score(y_incorrect[test_passed_mask], scores_arr[test_passed_mask])
            except ValueError:
                auroc_pass = float('nan')
        else:
            auroc_pass = float('nan')
        print(f"{name:<35} {auroc_all:>12.3f} {auroc_pass:>20.3f}")
        table_rows.append({'method': name, 'auroc_all': auroc_all, 'auroc_test_passed': auroc_pass})
    print("=" * 70)

    # ─── Step 8: Save outputs ─────────────────────────────────────────────
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
    df_scores.to_csv(out / 'exp_d_scores.csv', index=False)

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(out / 'exp_d_table1.csv', index=False)
    with open(out / 'exp_d_table1.txt', 'w') as f:
        f.write(df_table.to_string(index=False))

    ops = {
        'fit_time_seconds': fit_time,
        'fit_peak_memory_mb': fit_peak_mem / 1024 / 1024,
        'score_time_seconds': score_time,
        'score_peak_memory_mb': score_peak_mem / 1024 / 1024,
        'per_trajectory_latency_ms': per_traj_latency * 1000,
        'n_train': len(train_groups), 'n_test': len(test_groups),
        'em_iters': args.em_iters,
        'spectral_radius_A': float(np.max(np.abs(np.linalg.eigvals(A)))),
    }
    with open(out / 'exp_d_operational.json', 'w') as f:
        json.dump(ops, f, indent=2)

    report = {
        'source': f'SWE-bench/SWE-smith-trajectories split={args.split}',
        'n_instances_requested': args.n_instances,
        'n_train': len(train_groups), 'n_test': len(test_groups),
        'table1': table_rows, 'operational': ops,
    }
    with open(out / 'exp_d_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nAll outputs saved to {out}/")
    print("Done.")


if __name__ == '__main__':
    main()