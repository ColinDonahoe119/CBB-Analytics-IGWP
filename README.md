# College Basketball In-Game Win Probability (IGWP) Model

A custom in-game win probability model for NCAA basketball, built on possession-level play-by-play data and benchmarked head-to-head against existing baseline models on a frozen test set.

## Overview

This project builds a nonlinear parametric win probability model that estimates a team's likelihood of winning at any point during a game, given the current score, time remaining, possession, and team strength ratings. The model is evaluated against two existing baseline models (`winProb` and `bracketProb`) using an apples-to-apples comparison on a held-out set of games. For overview of model evaluation, visualizations, and insights, refer to in_game_win_probability notebook. 

## Model Architecture

The model takes the form:

```
z = x1 * (lead ± poss_feat) / timeRem^y1 + x2 * (timeRem/40) * spread / timeRem^y2
prob = sigmoid(z)
```

Where:
- **lead** — current point differential (favorite − underdog)
- **poss_feat** — possession adjustment based on team offensive/defensive ratings
- **timeRem** — minutes remaining in the game
- **spread** — pre-game team strength differential (net rating based)
- **x1, x2, y1, y2** — trainable parameters, including exponents, making the model nonlinear in its parameters

Because two of the parameters (`y1`, `y2`) are exponents rather than linear coefficients, the model is fit using `scipy.optimize` rather than standard logistic regression.

## Pipeline

| Script | Purpose |
|---|---|
| `data_prep.py` | One-time script joining possession-level data with team ratings/static team metadata and transforms columns to be used by model; produces train/test CSVs |
| `train_model.py` | Fits model parameters (`x1`, `x2`, `y1`, `y2`) via nonlinear optimization on the training set |
| `ingame_winprob.py` | Importable module with the trained model's inference function, plus baseline model implementations for comparison |
| `game_eval.py` | Computes win probability for every possession in a given game, across all models, and assembles game-level results |
| `model_eval.py` | Evaluation suite: Brier score, log loss, calibration MAE/RMSE, and calibration visualizations |
| `win_prob_chart.py` | Plots a single game's win probability trajectory over time |
| `in_game_win_probability.ipynb` |All in one notebook: Trains, evaluates, visuals, insights |


## Evaluation Methodology

Model performance is measured on a **frozen 100-game test set**, using the same split as the baseline models for direct comparability. Four metrics are used:

- **Brier Score** — mean squared error between predicted probability and actual outcome
- **Log Loss** — penalizes confident wrong predictions more heavily than Brier score
- **Calibration MAE / RMSE** — measures how closely predicted probabilities match actual outcome rates across probability bins (e.g., among all possessions predicted at ~70% win probability, do favorites actually win ~70% of the time?)

## Results

| Metric | Custom Model | Baseline Model |
|---|---|---|
| Brier Score | 0.117291 | 0.119241 |
| Log Loss | 0.364188 | 0.371980 |
| Cal MAE | 0.011116 | 0.043854 |
| Cal RMSE | 0.014380 | 0.053321 |

The custom model outperforms the baseline on all four metrics. The improvement in discrimination (Brier score, log loss) is modest, but the improvement in calibration is substantial — roughly a 4x reduction in calibration error. This means that when the custom model predicts a given win probability, that probability is a much more reliable estimate of the true likelihood of winning, which matters for any downstream use of the probabilities themselves (e.g., broadcast graphics, betting markets, decision support) rather than just rank-ordering likely winners.

## Additional Analysis

Beyond aggregate model evaluation, the project includes game-level analysis tools for identifying dramatic games:

- **comebacks** — for each game, finds the possession where the eventual winning team's win probability was at its lowest, identifying how close they came to losing
- **swings** — for each game, finds the single possession with the largest change in win probability from the possession immediately before it

## Tools

- Python (pandas, numpy, scipy, matplotlib)
- Google Colab for analysis and model training
- Google Drive for data sync

## Notes on Data

- Possession-level data (`possessions.csv`) is possession-first oriented — team strength ratings (`netRtg`, `ortg`, `drtg`, `pace`) are joined per-team, per-game-date, and scores are normalized so the favorite (determined by pre-game team rating) is always in the same column position across possessions.
- Evaluation strictly uses the frozen ~800-game test set; a separate 3200-game set is reserved for training only, to avoid any leakage between model fitting and benchmark comparison.
