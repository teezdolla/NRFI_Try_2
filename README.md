# NRFI Model Pipeline

This project predicts whether a run will be scored in the first inning (YRFI/NRFI).

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

### Features
- Leak-free rolling averages for pitchers and offense computed with `.shift().rolling()` to avoid data leakage.
- Pitcher form metrics including `K_pct`, `BB_pct` and rolling xERA.
- Offense metrics such as `OBP_team`, `SLG_team`, `OPS_team`, `ISO_team`, and an approximate `wOBA_team`.
- Support for ballpark factors and optional weather data.
- Time-series cross validation evaluates both calibrated logistic regression and a tuned XGBoost model.

### Training
`nrfi_pipeline.py` loads `final_training_data_clean_final.csv`, builds leak-free features and evaluates models with time-series cross-validation. Both logistic regression and XGBoost metrics are printed before a tuned XGBoost model is saved.

Run:
```bash
python nrfi_pipeline.py
```
### Dataset Balance
`final_training_data_clean_final.csv` holds 4,344 labeled half-innings. A label of `1` represents a YRFI (a run scored) while `0` means NRFI. The file contains 2,518 YRFI rows (about 58%) and 1,826 NRFI rows (about 42%).


### Daily Predictions

After installing the requirements, run:

```bash
python predict_today.py
```

The script multiplies the predicted probabilities for the top and bottom halves of the first inning, effectively assuming they are independent events. This means `P_YRFI_total` can be higher than either half on its own.

`predict_today.py` fetches the daily games and outputs probabilities for each half
of the first inning as well as the combined total for the entire inning. The
results are written to `predictions.txt`.
