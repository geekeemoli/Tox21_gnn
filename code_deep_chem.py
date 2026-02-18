from load_data import load_tox21_org
import deepchem as dc
import numpy as np

print("Loading and featurizing data")
#define the featurizer. We chose ConvMolFeaturizer as it is suitable for graph convolutional networks
#it represents molecules as graphs. Atoms = nodes, bonds = edges
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_partial_charge = True,use_chirality = True)

#load the tox21 dataset with the deepchem
tasks, transformers, train_dataset, valid_dataset, test_dataset = load_tox21_org(featurizer = featurizer)
#tasks: ['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', ...] -> list of toxicity assays that we try to predict. Hence, we need 12 output neurons in the final layer of our model
#datasets: This is a tuple (train, valid, test) containing 3 DiskDataset objects, designed to handle large datasets that may not fit entirely into memory
#Transformers: preprocessing steps applied to the data, such as normalization or standardization of features

print("Number of tasks: ", len(tasks))
for i, task in enumerate(tasks):
    print(f"Task {i+1}: {task}")
print("Number of training samples: ", len(train_dataset))

#Detect the number of atom features from the first training sample
number_atom_features = train_dataset.X[0].node_features.shape[1]
print(f"Number of atom features: {number_atom_features}")

#Define the model
model = dc.models.GCNModel(
    n_tasks=12,
    mode='classification',
    number_atom_features=number_atom_features,
    dropout=0.2,
    batch_size=32,
    learning_rate=0.001
)
print("Training GCN model")
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
