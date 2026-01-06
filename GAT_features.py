import os
import deepchem as dc
import numpy as np
from sklearn.metrics import roc_auc_score

print("Loading and featurizing data")
#define the featurizer. We chose ConvMolFeaturizer as it is suitable for graph convolutional networks
#it represents molecules as graphs. Atoms = nodes, bonds = edges
featurizer = dc.feat.MolGraphConvFeaturizer(use_partial_charge=True, use_edges=True) #use_edges=True to include bond information, additionaly chirality and partial charge attributes could be used

#load the tox21 dataset with the deepchem
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer) #SMILES strings into graph representations (objects that deepchem can work with)
train_dataset, valid_dataset, test_dataset = datasets #"Scaffold Split" to split the data into training, validation, and test sets based on molecular scaffolds
#tasks: ['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', ...] -> list of toxicity assays that we try to predict. Hence, we need 12 output neurons in the final layer of our model
#datasets: This is a tuple (train, valid, test) containing 3 DiskDataset objects, designed to handle large datasets that may not fit entirely into memory
#Transformers: preprocessing steps applied to the data, such as normalization or standardization of features

print("Number of tasks: ", len(tasks))
for i, task in enumerate(tasks):
    print(f"Task {i+1}: {task}")
print("Number of training samples: ", len(train_dataset))

#Define the model
"""
model = dc.models.GraphConvModel(
    n_tasks=12, 
    mode='classification', 
    dropout=0.2,
    batch_size=32,
    learning_rate=0.001
)
"""
for X, y, w, ids in train_dataset.itersamples():
    # X is a GraphData object. node_features is the matrix of atom details.
    n_features_detected = X.node_features.shape[1]
    print(f"Detected number of features per atom: {n_features_detected}")
    break  # We only need one sample to know the shape

model = dc.models.GATModel(
    n_tasks=12,
    number_atom_features=n_features_detected,
    mode='classification',
    dropout=0.2,
    batch_size=32,
    learning_rate=0.001,
    device='cpu'
)

print("Training GAT model")
#Train the model
model.fit(train_dataset, nb_epoch=10)
print("Evaluating model performance")
metric_per_task = dc.metrics.Metric(
    dc.metrics.roc_auc_score,
    mode="classification"
)
avg_scores, per_task_scores = model.evaluate(test_dataset, [metric_per_task], transformers, per_task_metrics=True)

auc_list = per_task_scores['roc_auc_score']
if isinstance(auc_list, float) or isinstance(auc_list, np.float64):
    auc_list = [auc_list]

for task_name, score in zip(tasks, auc_list):
    print(f"{task_name:15s}: {score:.4f}")

# Calculate mean AUC across all tasks
print(f"Average AUC    : {np.mean(auc_list):.4f}")