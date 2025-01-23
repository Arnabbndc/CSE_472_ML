# Assignment 4: PCA & EM Algorithm

## Overview

This assignment focuses on implementing Principal Component Analysis (PCA) and the Expectation-Maximization (EM) algorithm. The goal is to reduce the dimensionality of the dataset using PCA and to cluster the data using the EM algorithm.

### Directory Structure

```
CSE_472_ML/
    4_PCA & EM Algorithm/
        1905065_em.py
        1905065_pca.py
        em_data.txt
        pca_data.txt
```

### Files

- **1905065_em.py**: Python script implementing the EM algorithm for clustering.
- **1905065_pca.py**: Python script implementing PCA for dimensionality reduction.
- **em_data.txt**: Dataset used for the EM algorithm.
- **pca_data.txt**: Dataset used for PCA.

### Principal Component Analysis (PCA)

1. **Data Loading**:
    - Load the dataset from `pca_data.txt`.

2. **PCA Implementation**:
    - Standardize the data.
    - Compute the covariance matrix.
    - Calculate eigenvalues and eigenvectors.
    - Select the top principal components.
    - Transform the data to the new subspace.

3. **Visualization**:
    - Visualize the transformed data in the new subspace.

### Expectation-Maximization (EM) Algorithm

1. **Data Loading**:
    - Load the dataset from `em_data.txt`.

2. **EM Algorithm Implementation**:
    - Initialize the parameters (means, covariances, and mixing coefficients).
    - E-step: Compute the responsibilities.
    - M-step: Update the parameters.
    - Iterate until convergence.

3. **Clustering**:
    - Assign data points to clusters based on the highest responsibility.

### How to Run

1. Run the PCA script:

    ```sh
    python 1905065_pca.py
    ```

2. Run the EM algorithm script:

    ```sh
    python 1905065_em.py
    ```

### Results

- The PCA script reduces the dimensionality of the dataset and visualizes the transformed data.
- The EM algorithm script clusters the data and outputs the cluster assignments.

### Conclusion

This assignment demonstrates the implementation of PCA for dimensionality reduction and the EM algorithm for clustering. The results show the effectiveness of these techniques in data analysis.

### References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NumPy Documentation](https://numpy.org/doc/)

---
