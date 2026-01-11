import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data directly (pandas handles the .gz compression automatically)
# Ensure the file 'tox21_labels_train.csv.gz' is in your working directory
file_path = 'tox21_labels_train.csv.gz' 
df = pd.read_csv(file_path, compression='gzip')

# 2. Inspect the structure
print("Data Shape:", df.shape)
print(df.head())

# 3. specific Visualization: Count of active toxicity cases per assay
# Drop the ID column (first column) to focus on the numeric labels
toxicity_data = df.drop(columns=[df.columns[0]]) 

# Calculate sum of '1's (active toxicity) per column
counts = toxicity_data.sum().sort_values(ascending=False)

# Create the plot
plt.figure(figsize=(10, 6))
counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Active Toxicity Labels')
plt.xlabel('Assay Name')
plt.ylabel('Count of Toxic Compounds')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()