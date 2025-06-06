# NRFI Model Pipeline

This project predicts whether a run will be scored in the first inning (YRFI/NRFI).

### Features
- Leak-free rolling averages for pitchers and team offense using `.shift().rolling()`.
- Pitcher form metrics including `K_pct` and `BB_pct`.
- Offense metrics such as `OBP_team`, `SLG_team`, `OPS_team`, `ISO_team`.
- Support for ballpark factors and weather data.

### Training
`nrfi_pipeline.py` loads `final_training_data_clean_final.csv`, builds features, and performs time-series cross-validation. Metrics are printed and an XGBoost model is saved.

Run:
```bash
python nrfi_pipeline.py
```
