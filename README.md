NSP-Monitor

A tool that watches an LLM agent while it works and flags when it starts drifting off track.

Status: research in progress. A paper is in preparation, so results may change.

What it does

Long, multi-step agent tasks (like fixing a bug in a codebase) can go wrong slowly without ever throwing an error. NSP-Monitor models an agent's run as a path through a state space, fits a dynamics matrix to it with EM, and measures how much the run drifts over time.

What's in this repo
Folder	What it covers	Status
Exp_A	Synthetic recovery: generate fake runs from known parameters, refit the model, and measure the error (MSE)	Preliminary
Exp_C	Mixing-rate analysis on image-refinement tests (small pilot, 5 tests)	Preliminary
Exp_D	Real SWE-agent trajectories from the SWE-bench/SWE-smith-trajectories dataset on HuggingFace	Final (for the "pr" run)
Experiment D: real agent trajectories

Data: the "tool" split of SWE-bench/SWE-smith-trajectories. Correct and incorrect labels come from the dataset's resolved column. I requested 2,000 instances and used 874 in the run (585 train, 289 test, seed 42, 40 EM iterations).

Task: detect incorrect patches. I compared NSP-Monitor's scores against two simple baselines, trajectory length and a discrete HMM.

Test-passed group: 56 of the 289 test instances (45 incorrect, 11 correct) whose last 3 steps look like a passing test in the tool output. This is a text check, not a real test result.

AUROC	All test instances (289)	Test-passed group (56)
NSP final drift	0.445	0.386
NSP drift slope	0.434	0.574
NSP drift per step	0.411	0.549
Trajectory length	0.579	0.429
Discrete HMM	0.433	0.592

Across all 289 test instances, trajectory length did best. Within the test-passed group, NSP drift slope and the HMM did better than trajectory length. That group is small (56), so treat it as a hint, not a firm result. This run's rho(A) is 0.623.

Experiment C: mixing-rate analysis (pilot)

Five image-refinement tests (Test_3, 4, 5, 7, 8). Test_3 and Test_4 used Google and stopped early at 5 and 11 iterations. Test_5, 7, and 8 used OpenAI. For each test, I fit the dynamics matrix to the first 3 iterations of 7 image metrics and compared its spectral radius (the "predicted mixing rate") to the observed drift. The result was Spearman rho = -0.800, p = 0.104, which is not statistically significant with only 5 tests. Predicted values shift slightly between runs because EM starts from random numbers and I did not set a seed, but the ranking stayed the same.

Experiment A: synthetic recovery

Latent dimension 4, trajectory length 50, sample sizes of 100, 500, 1000, and 2000, with 10 seeds each and 40 EM iterations (settings reduced for testing). In the regular condition the MSE stays around 0.05. In the rank-deficient condition (true matrix has rank 2 out of 4) it is in the thousands. Only the mean and standard deviation across seeds were saved.

Run it yourself
bash
# Create and turn on an isolated Python environment (Windows Command Prompt)
python -m venv venv
venv\Scripts\activate

# Install the packages the project needs
pip install -r requirements.txt

# Experiment A (synthetic recovery)
python run_exp_a.py

# Experiment C (mixing-rate analysis, overwrites its CSV files)
python run_exp_c.py

# Experiment D (downloads data, takes about 15 minutes, needs internet)
python run_exp_d_real.py --n-instances 2000 --split tool --task-type pr

Results are saved to paper_data, one folder above wherever you run the script. No API keys are needed or included.

Built with

Python, NumPy, SciPy, and scikit-learn.

Who made this

Anemka Lydia Legbara, working with Professor Raz Saremi at NYU.
