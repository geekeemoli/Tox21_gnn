from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from load_data import load_tox21
import deepchem as dc

n_estimators = 100

tasks, transformers, train_dataset, valid_dataset, test_dataset = load_tox21(featurizer =  dc.feat.CircularFingerprint(size = 2048, radius = 2))

X_train = train_dataset.X
y_train = train_dataset.y
w_train = train_dataset.w

X_test = test_dataset.X
y_test = test_dataset.y
w_test = test_dataset.w

auc_list = []

for i, task_name in enumerate(tasks):
    # Prepare training data
    valid_rows_train = w_train[:, i] > 0
    X_task_train = X_train[valid_rows_train]
    y_task_train = y_train[valid_rows_train, i]

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', n_jobs=3)
    rf.fit(X_task_train, y_task_train)

    # Evaluate on test data
    valid_rows_test = w_test[:, i] > 0
    X_task_test = X_test[valid_rows_test]
    y_task_test = y_test[valid_rows_test, i]

    # Predict probabilities
    y_pred_proba = rf.predict_proba(X_task_test)[:, 1]

    # Calculate AUC
    score = roc_auc_score(y_task_test, y_pred_proba)
    auc_list.append(score)

    print(f"{task_name:15s}: {score:.4f}")

# Calculate mean AUC across all tasks
print(f"Average AUC    : {np.mean(auc_list):.4f}")




