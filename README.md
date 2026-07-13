# CBB-Analytics-IGWP
Custom in-game win probability (IGWP) model for NCAA basketball, built on possession-level play-by-play data. Uses a nonlinear parametric sigmoid fit via scipy.optimize, evaluated head-to-head against existing baseline models on a frozen test set using Brier score, log loss, and calibration error.
