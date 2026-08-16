import pandas as pd

df = pd.read_csv('paper_data/exp_d_scores.csv')
print("Total test rows:", len(df))
print()
print("resolved value counts:")
print(df['resolved'].value_counts())
print()
if 'baseline_length' in df.columns:
    # a proxy check: how many trajectories look like they reached a test-pass state
    # by checking the raw features file for the last-step test_pass flag
    feat = pd.read_csv('paper_data/exp_d_features.csv')
    last_steps = feat.sort_values('step').groupby('instance').tail(1)
    print("Last-step f4 (test_pass) value counts across ALL instances:")
    print(last_steps['f4'].value_counts())
