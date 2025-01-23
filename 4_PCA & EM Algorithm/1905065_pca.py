# pca_implementation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from sklearn.manifold import TSNE

def load_pca_data(filename):
    data = np.loadtxt(filename)
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def perform_pca(X):
    cov_matrix = np.cov(X.T)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort  eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    # eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]    
    return eigenvectors

def plot_pca_scatter(X, eigenvectors):
    pc1 = X @ eigenvectors[:, 0]
    pc2 = X @ eigenvectors[:, 1]
    plt.figure(figsize=(10, 8))
    plt.scatter(pc1, pc2, alpha=0.5)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First Two Principal Components')
    plt.savefig('pca_scatter.png')
    # plt.show()
    plt.close()

def create_umap_plot(X):
    umap__ = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    pc = umap__.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pc[:, 0], pc[:, 1], alpha=0.5)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP Visualization')
    plt.savefig('umap_plot.png')
    # plt.show()
    plt.close()

def create_tsne_plot(X):
    tsne = TSNE(n_components=2, random_state=42)
    pc = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pc[:, 0], pc[:, 1], alpha=0.5)
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.title('t-SNE Visualization')
    plt.savefig('tsne_plot.png')
    # plt.show()
    plt.close()


def main():
    data = np.loadtxt("pca_data.txt")
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    eigenvectors = perform_pca(X)
    plot_pca_scatter(X, eigenvectors)
    create_umap_plot(X)
    create_tsne_plot(X)

if __name__ == "__main__":
    main()
