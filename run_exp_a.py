"""
Experiment A: Synthetic Recovery.
Generates trajectories from a known NSP, fits via EM, reports MSE vs N.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ---- Configuration ----
D = 4               # latent dimension
K = 50              # trajectory length
N_LIST = [100, 500, 1000, 2000]   # reduced for testing
N_SEEDS = 10                  # reduced for testing
EM_ITERS = 40                # reduced for testing

np.random.seed(42)

# Make sure output folder exists
Path('../paper_data').mkdir(parents=True, exist_ok=True)

def make_true_params(rank_deficient=False):
    """Make ground-truth NSP parameters."""
    if rank_deficient:
        U = np.random.randn(D, D // 2) * 0.5
        A = U @ U.T
    else:
        A = np.random.randn(D, D) * 0.3
    B = np.random.randn(D, 1) * 0.5
    C = np.eye(D) + np.random.randn(D, D) * 0.1
    Q = np.eye(D) * 0.1
    Sigma = np.eye(D) * 0.2
    return A, B, C, Q, Sigma

def generate_trajectories(N, A, B, C, Q, Sigma):
    """Sample N trajectories of length K from the true NSP."""
    trajs, rewards = [], []
    for _ in range(N):
        X = np.zeros((K, D)); O = np.zeros((K, D))
        R = np.random.randint(0, 2, size=K).astype(float)
        X[0] = np.random.randn(D) * 0.5
        for k in range(K - 1):
            noise = np.random.multivariate_normal(np.zeros(D), Q)
            X[k+1] = A @ X[k] + (B * R[k]).flatten() + noise
        for k in range(K):
            obs_noise = np.random.multivariate_normal(np.zeros(D), Sigma)
            O[k] = C @ X[k] + obs_noise
        trajs.append(O); rewards.append(R)
    return trajs, rewards

def kalman_smoother(O, R, A, B, C, Q, Sigma):
    """Forward-backward Kalman smoother for one trajectory."""
    T = len(O)
    mu_f = np.zeros((T, D)); P_f = np.zeros((T, D, D))
    P_f[0] = np.eye(D)
    for t in range(1, T):
        mu_pred = A @ mu_f[t-1] + (B * R[t-1]).flatten()
        P_pred = A @ P_f[t-1] @ A.T + Q
        S = C @ P_pred @ C.T + Sigma
        # Fix: use pinv instead of inv to handle singular matrices
        K_gain = P_pred @ C.T @ np.linalg.pinv(S + 1e-6 * np.eye(D))
        mu_f[t] = mu_pred + K_gain @ (O[t] - C @ mu_pred)
        P_f[t] = (np.eye(D) - K_gain @ C) @ P_pred
    mu_s = mu_f.copy(); P_s = P_f.copy()
    for t in range(T - 2, -1, -1):
        mu_pred = A @ mu_f[t] + (B * R[t]).flatten()
        P_pred = A @ P_f[t] @ A.T + Q
        J = P_f[t] @ A.T @ np.linalg.pinv(P_pred + 1e-6 * np.eye(D))
        mu_s[t] = mu_f[t] + J @ (mu_s[t+1] - mu_pred)
        P_s[t] = P_f[t] + J @ (P_s[t+1] - P_pred) @ J.T
    return mu_s, P_s

def fit_em(trajs, rewards, n_iters=EM_ITERS):
    """Fit A by EM (we only track A's MSE as the headline parameter)."""
    A = np.random.randn(D, D) * 0.1
    B = np.random.randn(D, 1) * 0.1
    C = np.eye(D); Q = np.eye(D) * 0.5; Sigma = np.eye(D) * 0.5
    for _ in range(n_iters):
        smoothed = [kalman_smoother(O, R, A, B, C, Q, Sigma)
                    for O, R in zip(trajs, rewards)]
        num = np.zeros((D, D)); den = np.zeros((D, D))
        for (mu_s, P_s), R in zip(smoothed, rewards):
            T = len(mu_s)
            for t in range(T - 1):
                num += np.outer(mu_s[t+1], mu_s[t])
                den += np.outer(mu_s[t], mu_s[t]) + P_s[t]
        A = num @ np.linalg.pinv(den + 1e-4 * np.eye(D))
    return A

def run(rank_deficient=False):
    A_true, B_true, C_true, Q_true, Sigma_true = make_true_params(rank_deficient)
    rows = []
    for N in N_LIST:
        mses = []
        for seed in tqdm(range(N_SEEDS),
                         desc=f'N={N}, rd={rank_deficient}'):
            np.random.seed(seed + 1000)
            trajs, rewards = generate_trajectories(
                N, A_true, B_true, C_true, Q_true, Sigma_true)
            A_hat = fit_em(trajs, rewards)
            mses.append(float(np.mean((A_hat - A_true) ** 2)))
        rows.append({'N': N, 'rank_deficient': rank_deficient,
                     'mse_mean': float(np.mean(mses)),
                     'mse_std': float(np.std(mses))})
    return rows

if __name__ == '__main__':
    print('Running regular condition (R1-R4 hold)...')
    reg = run(rank_deficient=False)
    # Save regular condition immediately
    df_reg = pd.DataFrame(reg)
    df_reg.to_csv('../paper_data/exp_a_results_regular.csv', index=False)
    print('Regular condition saved!')
    print(df_reg.to_string(index=False))

    print('\nRunning rank-deficient condition (R2 violated)...')
    rd = run(rank_deficient=True)
    # Save rank-deficient condition immediately
    df_rd = pd.DataFrame(rd)
    df_rd.to_csv('../paper_data/exp_a_results_rankdeficient.csv', index=False)
    print('Rank-deficient condition saved!')
    print(df_rd.to_string(index=False))

    # Save combined results
    df = pd.DataFrame(reg + rd)
    df.to_csv('../paper_data/exp_a_results.csv', index=False)
    print('\nFull results saved to paper_data/exp_a_results.csv')
    print(df.to_string(index=False))