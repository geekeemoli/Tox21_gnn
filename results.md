# Benchmark Random forest

## Deepchem dataset

Training RF with 400 estimators on 4096-bit fingerprints (radius 2, chiral=True)
NR-AR          : 0.7140
NR-AR-LBD      : 0.8460
NR-AhR         : 0.8166
NR-Aromatase   : 0.7118
NR-ER          : 0.6210
NR-ER-LBD      : 0.5997
NR-PPAR-gamma  : 0.7652
SR-ARE         : 0.6784
SR-ATAD5       : 0.7219
SR-HSE         : 0.7142
SR-MMP         : 0.8048
SR-p53         : 0.7548
Average AUC    : 0.7290

## Using the original data set

Training RF with 400 estimators on 4096-bit fingerprints (radius 2, chiral=True)
NR-AhR         : 0.8304
NR-AR          : 0.4735
NR-AR-LBD      : 0.6145
NR-Aromatase   : 0.7744
NR-ER          : 0.6899
NR-ER-LBD      : 0.6319
NR-PPAR-gamma  : 0.7015
SR-ARE         : 0.7862
SR-ATAD5       : 0.7605
SR-HSE         : 0.7879
SR-MMP         : 0.8624
SR-p53         : 0.6755
Average AUC    : 0.7157

# GAT model: finding the right feature set

# No features

Number of training samples:  6245
Detected number of features per atom: 30
Training GAT model
Features: Partial Charge: False, Edges: False, Chirality: False
Evaluating model performance
NR-AR          : 0.7252
NR-AR-LBD      : 0.8125
NR-AhR         : 0.7491
NR-Aromatase   : 0.6817
NR-ER          : 0.6722
NR-ER-LBD      : 0.6924
NR-PPAR-gamma  : 0.7366
SR-ARE         : 0.6150
SR-ATAD5       : 0.7060
SR-HSE         : 0.7038
SR-MMP         : 0.7516
SR-p53         : 0.6804
Average AUC    : 0.7105

Number of training samples:  11761
Detected number of features per atom: 30
Training GAT model
Features: Partial Charge: False, Edges: False, Chirality: False
Evaluating model performance
NR-AhR         : 0.8662
NR-AR          : 0.2435
NR-AR-LBD      : 0.6670
NR-Aromatase   : 0.7180
NR-ER          : 0.6606
NR-ER-LBD      : 0.7077
NR-PPAR-gamma  : 0.6384
SR-ARE         : 0.7914
SR-ATAD5       : 0.6009
SR-HSE         : 0.7294
SR-MMP         : 0.7995
SR-p53         : 0.7544
Average AUC    : 0.6814

## Partial charge

Number of training samples:  6245
Detected number of features per atom: 31
Training GAT model
Features: Partial Charge: True, Edges: False, Chirality: False
Evaluating model performance
NR-AR          : 0.7095
NR-AR-LBD      : 0.7556
NR-AhR         : 0.7595
NR-Aromatase   : 0.6432
NR-ER          : 0.6384
NR-ER-LBD      : 0.6603
NR-PPAR-gamma  : 0.6367
SR-ARE         : 0.5663
SR-ATAD5       : 0.6889
SR-HSE         : 0.6674
SR-MMP         : 0.7505
SR-p53         : 0.6665
Average AUC    : 0.6786


Number of training samples:  11761
Detected number of features per atom: 31
Training GAT model
Features: Partial Charge: True, Edges: False, Chirality: False
Evaluating model performance
NR-AhR         : 0.8548
NR-AR          : 0.2378
NR-AR-LBD      : 0.7800
NR-Aromatase   : 0.6735
NR-ER          : 0.6452
NR-ER-LBD      : 0.6969
NR-PPAR-gamma  : 0.6508
SR-ARE         : 0.7538
SR-ATAD5       : 0.6703
SR-HSE         : 0.7245
SR-MMP         : 0.7706
SR-p53         : 0.7030
Average AUC    : 0.6801

# edges

Number of training samples:  6243
Detected number of features per atom: 30
Training GAT model
Features: Partial Charge: False, Edges: True, Chirality: False
Evaluating model performance
NR-AR          : 0.7656
NR-AR-LBD      : 0.7634
NR-AhR         : 0.7622
NR-Aromatase   : 0.6655
NR-ER          : 0.6610
NR-ER-LBD      : 0.6833
NR-PPAR-gamma  : 0.7215
SR-ARE         : 0.6113
SR-ATAD5       : 0.6983
SR-HSE         : 0.7144
SR-MMP         : 0.7415
SR-p53         : 0.6475
Average AUC    : 0.7030

Number of training samples:  11748
Detected number of features per atom: 30
Training GAT model
Features: Partial Charge: False, Edges: True, Chirality: False
Evaluating model performance
NR-AhR         : 0.8397
NR-AR          : 0.2275
NR-AR-LBD      : 0.8193
NR-Aromatase   : 0.6875
NR-ER          : 0.6642
NR-ER-LBD      : 0.7161
NR-PPAR-gamma  : 0.6456
SR-ARE         : 0.7615
SR-ATAD5       : 0.6159
SR-HSE         : 0.7287
SR-MMP         : 0.7669
SR-p53         : 0.7160
Average AUC    : 0.6824

# Chirality


Number of training samples:  6245
Detected number of features per atom: 32
Training GAT model
Features: Partial Charge: False, Edges: False, Chirality: True
Evaluating model performance
NR-AR          : 0.7861
NR-AR-LBD      : 0.7758
NR-AhR         : 0.7553
NR-Aromatase   : 0.6293
NR-ER          : 0.6537
NR-ER-LBD      : 0.6588
NR-PPAR-gamma  : 0.6560
SR-ARE         : 0.5931
SR-ATAD5       : 0.7082
SR-HSE         : 0.6771
SR-MMP         : 0.7149
SR-p53         : 0.6815
Average AUC    : 0.6908


Number of training samples:  11761
Detected number of features per atom: 32
Training GAT model
Features: Partial Charge: False, Edges: False, Chirality: True
Evaluating model performance
NR-AhR         : 0.8665
NR-AR          : 0.2662
NR-AR-LBD      : 0.7774
NR-Aromatase   : 0.6974
NR-ER          : 0.6621
NR-ER-LBD      : 0.6878
NR-PPAR-gamma  : 0.6372
SR-ARE         : 0.7925
SR-ATAD5       : 0.5553
SR-HSE         : 0.6797
SR-MMP         : 0.7624
SR-p53         : 0.7151
Average AUC    : 0.6750

# Partial Charge, Edges

Number of training samples:  6243
Detected number of features per atom: 31
Training GAT model
Features: Partial Charge: True, Edges: True, Chirality: False
Evaluating model performance
NR-AR          : 0.7375
NR-AR-LBD      : 0.7770
NR-AhR         : 0.7685
NR-Aromatase   : 0.6524
NR-ER          : 0.6713
NR-ER-LBD      : 0.6082
NR-PPAR-gamma  : 0.6920
SR-ARE         : 0.6152
SR-ATAD5       : 0.6992
SR-HSE         : 0.7030
SR-MMP         : 0.7506
SR-p53         : 0.6786
Average AUC    : 0.6961

Number of training samples:  11748
Detected number of features per atom: 31
Training GAT model
Features: Partial Charge: True, Edges: True, Chirality: False
Evaluating model performance
NR-AhR         : 0.8436
NR-AR          : 0.2662
NR-AR-LBD      : 0.8682
NR-Aromatase   : 0.6880
NR-ER          : 0.6926
NR-ER-LBD      : 0.7878
NR-PPAR-gamma  : 0.6007
SR-ARE         : 0.7959
SR-ATAD5       : 0.6238
SR-HSE         : 0.6895
SR-MMP         : 0.7546
SR-p53         : 0.7488
Average AUC    : 0.6966

# Attentive FP

## All features, deepchem data

Evaluating model performance
NR-AR          : 0.6490
NR-AR-LBD      : 0.7861
NR-AhR         : 0.7985
NR-Aromatase   : 0.6867
NR-ER          : 0.7023
NR-ER-LBD      : 0.7906
NR-PPAR-gamma  : 0.6529
SR-ARE         : 0.6593
SR-ATAD5       : 0.6947
SR-HSE         : 0.7706
SR-MMP         : 0.7948
SR-p53         : 0.6987
Average AUC    : 0.7237

## All features, original data

Evaluating model performance
NR-AhR         : 0.8409
NR-AR          : 0.6871
NR-AR-LBD      : 0.8279
NR-Aromatase   : 0.7440
NR-ER          : 0.5838
NR-ER-LBD      : 0.7045
NR-PPAR-gamma  : 0.7606
SR-ARE         : 0.7636
SR-ATAD5       : 0.7061
SR-HSE         : 0.8234
SR-MMP         : 0.8442
SR-p53         : 0.7619
Average AUC    : 0.7540

Later version, with matching ROC (10 epochs):
NR-AhR         : 0.8948
NR-AR          : 0.4494
NR-AR-LBD      : 0.8373
NR-Aromatase   : 0.7686
NR-ER          : 0.5980
NR-ER-LBD      : 0.6685
NR-PPAR-gamma  : 0.6989
SR-ARE         : 0.8086
SR-ATAD5       : 0.6893
SR-HSE         : 0.8591
SR-MMP         : 0.8356
SR-p53         : 0.7735
Average AUC    : 0.7401

# AttentiveFP ensemble

# deepchem data, all features

--- Ensemble Results ---
NR-AR          : 0.7063
NR-AR-LBD      : 0.7631
NR-AhR         : 0.7997
NR-Aromatase   : 0.7105
NR-ER          : 0.7128
NR-ER-LBD      : 0.7967
NR-PPAR-gamma  : 0.7270
SR-ARE         : 0.6753
SR-ATAD5       : 0.7351
SR-HSE         : 0.7933
SR-MMP         : 0.7891
SR-p53         : 0.6972

Ensemble Average AUC : 0.7422

# origianl data, all features

--- Ensemble Results ---
NR-AhR         : 0.8396
NR-AR          : 0.5529
NR-AR-LBD      : 0.8562
NR-Aromatase   : 0.7602
NR-ER          : 0.5958
NR-ER-LBD      : 0.6399
NR-PPAR-gamma  : 0.7777
SR-ARE         : 0.8046
SR-ATAD5       : 0.7287
SR-HSE         : 0.8014
SR-MMP         : 0.8473
SR-p53         : 0.7595

Ensemble Average AUC : 0.7470


New version:
--- Ensemble Results ---
NR-AhR         : 0.8828
NR-AR          : 0.4767
NR-AR-LBD      : 0.8553
NR-Aromatase   : 0.7480
NR-ER          : 0.5937
NR-ER-LBD      : 0.6514
NR-PPAR-gamma  : 0.7302
SR-ARE         : 0.8283
SR-ATAD5       : 0.7178
SR-HSE         : 0.8301
SR-MMP         : 0.8283
SR-p53         : 0.7845

Ensemble Average AUC : 0.7439

# GCN (all features)
## Deepchem dataset

Number of training samples:  6243
Number of atom features: 33
Training GCN model
Evaluating model performance
NR-AR          : 0.7305
NR-AR-LBD      : 0.7944
NR-AhR         : 0.7904
NR-Aromatase   : 0.6856
NR-ER          : 0.6753
NR-ER-LBD      : 0.6617
NR-PPAR-gamma  : 0.6904
SR-ARE         : 0.6202
SR-ATAD5       : 0.7203
SR-HSE         : 0.7369
SR-MMP         : 0.7585
SR-p53         : 0.6941
Average AUC    : 0.7132

## Original dataset

Number of training samples:  11748
Number of atom features: 33
Training GCN model
Evaluating model performance
NR-AhR         : 0.8540
NR-AR          : 0.3902
NR-AR-LBD      : 0.8613
NR-Aromatase   : 0.7218
NR-ER          : 0.6310
NR-ER-LBD      : 0.7682
NR-PPAR-gamma  : 0.6517
SR-ARE         : 0.8031
SR-ATAD5       : 0.6511
SR-HSE         : 0.8360
SR-MMP         : 0.8249
SR-p53         : 0.7563
Average AUC    : 0.7291
