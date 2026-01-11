import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the "Dense" features (The actual learning data)
# This file contains calculated properties like Molecular Weight, Solubility, etc.
feature_file = 'tox21_dense_train.csv.gz'
df_features = pd.read_csv(feature_file, compression='gzip')

# 2. Inspect the structure
print("Feature Matrix Shape:", df_features.shape)
print("First 5 rows of features:")
print(df_features.head())

# 3. Visualize a specific feature (e.g., Molecular Weight 'AWeight')
# This helps you see the distribution of the chemical properties in your dataset
plt.figure(figsize=(10, 6))
if 'WPSA1' in df_features.columns:
    df_features['WPSA1'].hist(bins=50, color='teal', edgecolor='black')
    plt.title('Distribution of WPSA1')
    plt.xlabel('WPSA1')
    plt.ylabel('Count')
else:
    # Fallback to the 2nd column if AWeight isn't there
    df_features.iloc[:, 1].hist(bins=50, color='teal', edgecolor='black')
    plt.title('Distribution of Feature: ' + df_features.columns[1])

plt.tight_layout()
plt.show()