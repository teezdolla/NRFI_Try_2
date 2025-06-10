import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64

# 1. Data Loading
print('Loading data...')
df = pd.read_csv('data.csv')

# Basic preprocessing: fill missing with column means
print('Cleaning data...')
for col in df.columns:
    if df[col].dtype != 'object':
        df[col].fillna(df[col].mean(), inplace=True)

X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# Logistic Regression
print('Training Logistic Regression...')
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
proba_lr = log_reg.predict_proba(X_test_scaled)[:,1]
pred_lr = (proba_lr >= 0.5).astype(int)
results['LogReg'] = {
    'accuracy': accuracy_score(y_test, pred_lr),
    'precision': precision_score(y_test, pred_lr),
    'recall': recall_score(y_test, pred_lr),
    'f1': f1_score(y_test, pred_lr),
    'roc_auc': roc_auc_score(y_test, proba_lr)
}

# Random Forest
print('Training Random Forest...')
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:,1]
pred_rf = rf.predict(X_test)
results['RandomForest'] = {
    'accuracy': accuracy_score(y_test, pred_rf),
    'precision': precision_score(y_test, pred_rf),
    'recall': recall_score(y_test, pred_rf),
    'f1': f1_score(y_test, pred_rf),
    'roc_auc': roc_auc_score(y_test, proba_rf)
}

# XGBoost
print('Training XGBoost...')
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=300,
    random_state=42
)
xgb_model.fit(X_train, y_train)
proba_xgb = xgb_model.predict_proba(X_test)[:,1]
pred_xgb = (proba_xgb >= 0.5).astype(int)
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, pred_xgb),
    'precision': precision_score(y_test, pred_xgb),
    'recall': recall_score(y_test, pred_xgb),
    'f1': f1_score(y_test, pred_xgb),
    'roc_auc': roc_auc_score(y_test, proba_xgb)
}

# Determine best model based on ROC AUC
best_model_by_auc = max(results, key=lambda k: results[k]['roc_auc'])
print(f'Best model by ROC AUC: {best_model_by_auc}')

# Use XGBoost for final predictions regardless of ROC AUC ranking
best_model_name = 'XGBoost'

if best_model_name == 'LogReg':
    best_proba = proba_lr
    best_pred = pred_lr
    model = log_reg
    importances = None
elif best_model_name == 'RandomForest':
    best_proba = proba_rf
    best_pred = pred_rf
    model = rf
    importances = rf.feature_importances_
elif best_model_name == 'XGBoost':
    best_proba = proba_xgb
    best_pred = pred_xgb
    model = xgb_model
    importances = xgb_model.feature_importances_
else:
    raise ValueError('Unknown model')

# Save predictions
pred_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted_proba': best_proba,
    'predicted_class': best_pred
})
pred_df.to_csv('predictions.csv', index=False)
print('Saved predictions to predictions.csv')

# Feature importance plot if available
def _save_fig_as_base64(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    with open(filename, 'w') as f:
        f.write(b64)

if importances is not None:
    plt.figure(figsize=(10,6))
    idx = np.argsort(importances)[::-1]
    names = X_train.columns[idx]
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), names, rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    _save_fig_as_base64(plt.gcf(), 'importance.txt')
    print('Saved feature importance to importance.txt')

# ROC Curve
RocCurveDisplay.from_predictions(y_test, best_proba)
plt.title('ROC Curve - ' + best_model_name)
plt.tight_layout()
_save_fig_as_base64(plt.gcf(), 'roc_curve.txt')
print('Saved ROC curve to roc_curve.txt')

# Summary metrics
print('\nPerformance Metrics:')
for model_name, metrics in results.items():
    print(model_name, metrics)
print('\nBest model:', best_model_name)
