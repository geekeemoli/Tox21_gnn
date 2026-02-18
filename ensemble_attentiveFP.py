import deepchem as dc
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import torch
from load_data import load_tox21_org

print("Loading and featurizing data...")
featurizer = dc.feat.MolGraphConvFeaturizer(
    use_edges=True, 
    use_partial_charge=True, 
    use_chirality=True
)
tasks, transformers, train_dataset, valid_dataset, test_dataset = load_tox21_org(featurizer = featurizer)
print(f"Number of tasks: {len(tasks)}")
print(f"Number of training samples: {len(train_dataset)}")

for X, y, w, ids in train_dataset.itersamples():
    n_features_detected = X.node_features.shape[1]
    print(f"Detected features per atom: {n_features_detected}")
    break

N_ENSEMBLE = 3  # Number of models in the ensemble
ensemble_models = []

print(f"\nStarting training for Ensemble of {N_ENSEMBLE} models...")

# 3. Training Loop
for i in range(N_ENSEMBLE):
    print(f"\n--- Training Model {i+1}/{N_ENSEMBLE} ---")
    
    # Optional: Set a different seed for each model to ensure diversity
    # (Though random initialization usually handles this, explicit seeding is safer)
    torch.manual_seed(42 + i)
    np.random.seed(42 + i)

    model = dc.models.AttentiveFPModel(
        n_tasks=12,
        number_atom_features=n_features_detected,
        mode='classification',
        dropout=0.2,
        batch_size=128,
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Fit the individual model
    # Note: We use restore=False to ensure we start from scratch every time
    model.fit(train_dataset, nb_epoch=10, restore=False)
    
    ensemble_models.append(model)

print("\nAll models trained. Calculating Ensemble predictions...")

# 4. Aggregate Predictions (Soft Voting)
# We need to collect probabilities from each model.
# DeepChem predict returns shape: (n_samples, n_tasks, n_classes)
# For TOX21, n_classes is 2.
predictions_list = []

for i, model in enumerate(ensemble_models):
    print(f"Generating predictions for Model {i+1}...")
    # predict() returns the probabilities
    pred = model.predict(test_dataset) 
    predictions_list.append(pred)

# Stack along a new axis to get shape (n_models, n_samples, n_tasks, n_classes)
all_preds = np.array(predictions_list)

# Take the MEAN across the 0th axis (the models)
# This averages the probabilities for class 0 and class 1
ensemble_probs = np.mean(all_preds, axis=0)

# 5. Evaluate the Ensemble
print("\nEvaluating Ensemble Performance...")

# We use DeepChem's Metric class but call compute_metric manually
# so we can pass our custom averaged probabilities.
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, 
    mode="classification"
)

# test_dataset.y contains true labels
# test_dataset.w contains weights (handling missing data)
# ensemble_probs contains our averaged predictions
combined_score = metric.compute_metric(
    test_dataset.y,
    ensemble_probs,
    test_dataset.w,
    per_task_metrics=True
)

# The result is a tuple (average_score, dict_of_scores) because per_task_metrics=True
avg_auc = combined_score[0]
per_task_auc = combined_score[1]

print("\n--- Ensemble Results ---")
for task_name, score in zip(tasks, per_task_auc):
    print(f"{task_name:15s}: {score:.4f}")

print(f"\nEnsemble Average AUC : {avg_auc:.4f}")


y_true = test_dataset.y
w = test_dataset.w
predictions = ensemble_probs

# Plot ROC curves
print("\nGenerating ROC curves...")

# Plot ROC curve for each task
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for task_idx, task_name in enumerate(tasks):
    mask = w[:, task_idx] > 0
    fpr, tpr, _ = roc_curve(y_true[mask, task_idx], predictions[mask, task_idx, 1])
    roc_auc = auc(fpr, tpr)
    
    axes[task_idx].plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[task_idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[task_idx].set_xlim([0.0, 1.0])
    axes[task_idx].set_ylim([0.0, 1.05])
    axes[task_idx].set_xlabel('False Positive Rate')
    axes[task_idx].set_ylabel('True Positive Rate')
    axes[task_idx].set_title(f'{task_name}')
    axes[task_idx].legend(loc="lower right")
    axes[task_idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
print("ROC curves saved as 'roc_curves.png'")
