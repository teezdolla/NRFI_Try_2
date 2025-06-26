import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
)
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64

print('Loading data...')
df = pd.read_csv('data.csv')

print('Cleaning data...')
for col in df.columns:
    if df[col].dtype != 'object':
        df[col].fillna(df[col].mean(), inplace=True)

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Training XGBoost...')
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=300,
    random_state=42,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    'accuracy': accuracy_score(y_test, pred),
    'precision': precision_score(y_test, pred),
    'recall': recall_score(y_test, pred),
    'f1': f1_score(y_test, pred),
    'roc_auc': roc_auc_score(y_test, proba),
}

pred_df = pd.DataFrame(
    {'actual': y_test.values, 'predicted_proba': proba, 'predicted_class': pred}
)
pred_df.to_csv('predictions.csv', index=False)
print('Saved predictions to predictions.csv')

importances = model.feature_importances_
plt.figure(figsize=(10, 6))
idx = np.argsort(importances)[::-1]
names = X_train.columns[idx]
plt.bar(range(len(importances)), importances[idx])
plt.xticks(range(len(importances)), names, rotation=90)
plt.title('Feature Importance')
plt.tight_layout()

def _save_fig_as_base64(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    with open(filename, 'w') as f:
        f.write(b64)

_save_fig_as_base64(plt.gcf(), 'importance.txt')
print('Saved feature importance to importance.txt')

RocCurveDisplay.from_predictions(y_test, proba)
plt.title('ROC Curve - XGBoost')
plt.tight_layout()
_save_fig_as_base64(plt.gcf(), 'roc_curve.txt')
print('Saved ROC curve to roc_curve.txt')

print('\nPerformance Metrics:')
print(metrics)
