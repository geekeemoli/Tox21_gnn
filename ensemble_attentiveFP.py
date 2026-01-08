import os
import deepchem as dc
import numpy as np
import torch

print("Loading and featurizing data...")
featurizer = dc.feat.MolGraphConvFeaturizer(
    use_edges=True, 
    use_partial_charge=True, 
    use_chirality=True
)

tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer)
train_dataset, valid_dataset, test_dataset = datasets

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
        batch_size=32,
        learning_rate=0.001,
        device='cpu' # Warning: CPU training is slow for ensembles
    )
    
    # Fit the individual model
    # Note: We use restore=False to ensure we start from scratch every time
    model.fit(train_dataset, nb_epoch=20, restore=False)
    
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