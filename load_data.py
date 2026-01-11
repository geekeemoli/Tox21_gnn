import deepchem as dc

def load_tox21(featurizer = dc.feat.MolGraphConvFeaturizer(use_edges = True)):
    print("Loading and featurizing data")

    #load the tox21 dataset with the deepchem
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer) #SMILES strings into graph representations (objects that deepchem can work with)
    train_dataset, valid_dataset, test_dataset = datasets #"Scaffold Split" to split the data into training, validation, and test sets based on molecular scaffolds

    return tasks, transformers, train_dataset, valid_dataset, test_dataset
