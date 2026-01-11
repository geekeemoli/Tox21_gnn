import scipy.io
import matplotlib.pyplot as plt
import gzip

# Update this path to where your file is located
file_path = 'tox21_sparse_train.mtx.gz'

print(f"--- Inspecting Sparse Matrix: {file_path} ---")

try:
    # 1. Open and Load the Matrix
    # We use gzip.open to handle the compression, and mmread to parse the matrix format
    with gzip.open(file_path, 'rb') as f:
        sparse_matrix = scipy.io.mmread(f)

    # Convert to CSR (Compressed Sparse Row) format for fast slicing and math
    sparse_matrix_csr = sparse_matrix.tocsr()

    # 2. Basic Structure Info
    n_compounds, n_features = sparse_matrix_csr.shape
    print(f"Dimensions: {n_compounds} Compounds x {n_features} Features")
    print(f"Total Non-Zero Entries: {sparse_matrix_csr.nnz}")
    
    # Calculate how 'full' the matrix is
    density = sparse_matrix_csr.nnz / (n_compounds * n_features)
    print(f"Sparsity/Density: {density:.4%} (Only this % of the matrix has data)")

    # 3. Peek at the Content
    # Since we can't print the whole matrix, we print the first few "active" entries
    print("\n--- Content Sample (First 10 Active Fingerprints) ---")
    rows, cols = sparse_matrix_csr.nonzero()
    for i in range(10):
        # This reads: "Compound [Row] has Feature [Col] active with value [Val]"
        val = sparse_matrix_csr[rows[i], cols[i]]
        print(f"Compound Index {rows[i]} -> Has Feature Index {cols[i]} (Value: {val})")

    # 4. Visualization (Spy Plot)
    # A 'Spy Plot' is the standard way to visualize sparse matrices. 
    # It places a black dot wherever there is a non-zero value.
    plt.figure(figsize=(10, 8))
    
    # We slice the top-left corner (e.g., first 500x500) to keep the plot readable
    subset_size = 500
    plt.spy(sparse_matrix_csr[:subset_size, :subset_size], markersize=1, color='black')
    
    plt.title(f'Visualizing the Data Structure (First {subset_size}x{subset_size})')
    plt.xlabel('Features (Chemical Fingerprints)')
    plt.ylabel('Compounds (Samples)')
    plt.show()

except Exception as e:
    print(f"Error reading the file: {e}")
    print("Ensure the file path is correct and scipy is installed (pip install scipy).")