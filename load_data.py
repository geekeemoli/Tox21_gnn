import deepchem as dc
from datasets import load_dataset
import numpy as np


def load_tox21(featurizer = dc.feat.MolGraphConvFeaturizer(use_edges = True), data_dir=None, save_dir=None):
    print("Loading and featurizing data")

    #load the tox21 dataset with the deepchem
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer, data_dir=data_dir, save_dir=save_dir) #SMILES strings into graph representations (objects that deepchem can work with)
    train_dataset, valid_dataset, test_dataset = datasets #"Scaffold Split" to split the data into training, validation, and test sets based on molecular scaffolds

    return tasks, transformers, train_dataset, valid_dataset, test_dataset

def load_tox21_org(featurizer = dc.feat.MolGraphConvFeaturizer(use_edges = True), data_dir = None, save_dir = None):
    ds = load_dataset("ml-jku/tox21")
    
    tasks = [
        'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    def make_dc_data(data):
        smiles = data["smiles"]
        features = featurizer.featurize(smiles)
        idx = [i for i, f in enumerate(features) if f is not None and getattr(f, "size", 1) > 0]
        features = features[idx]

        y = []
        w = []
        for i in idx:
            vals = data[i]
            row_y = []
            row_w = []
            for task in tasks:
                if vals[task] is None:
                    row_y.append(0.0)
                    row_w.append(0.0)
                else:
                    row_y.append(float(vals[task]))
                    row_w.append(1.0)
            y.append(row_y)
            w.append(row_w)

        return(dc.data.NumpyDataset(X = features, y = np.array(y), w = np.array(w), ids = smiles[idx]))



    train_dataset = make_dc_data(ds["train"])
    valid_dataset = make_dc_data(ds["validation"])
    test_dataset = valid_dataset

    balancer = dc.trans.BalancingTransformer(dataset=train_dataset)

    train_dataset = balancer.transform(train_dataset)
    valid_dataset = balancer.transform(valid_dataset)
    test_dataset = balancer.transform(test_dataset)
   
    # Add it to the list of transformers to be returned
    transformers = [balancer]

    return tasks, transformers, train_dataset, valid_dataset, test_dataset





