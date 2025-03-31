import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from matplotlib.colors import ListedColormap

# Setam stilul general pentru grafice
plt.style.use('ggplot')  # Folosim un stil diferit de default
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300  # Rezolutie inalta pentru grafice
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def create_enhanced_visualizations(companies_path="data/output/companies_with_insurance_labels.csv", 
                                 embeddings_path="data/output/company_embeddings.npy"):
    """Crează vizualizări îmbunătățite pentru datele companiilor și embeddings-urile acestora"""
    
    # Cream directorul pentru vizualizari
    os.makedirs("visualizations", exist_ok=True)
    
    # Incarcam datele
    print(" Loading data...")
    df = pd.read_csv(companies_path)  # Datele companiilor
    embeddings = np.load(embeddings_path)  # Embeddings-urile generate anterior
    
    # Aplicam PCA pentru reducerea dimensiunii
    pca = PCA(n_components=50)  # Reducem la 50 de componente principale
    embeddings_pca = pca.fit_transform(embeddings)  # Reducem dimensiunea embeddings-urilor
    
    # Aplicam t-SNE pentru o vizualizare 2D a datelor
    tsne = TSNE(n_components=2, random_state=42)  # Setam random_state pentru reproductibilitate
    tsne_embeddings = tsne.fit_transform(embeddings_pca)  # Reducem la 2 dimensiuni
    
    # Cream plotul cu t-SNE
    print("🔹 Visualizing data...")
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=df['insurance_label'].astype('category').cat.codes, cmap=ListedColormap(sns.color_palette("husl", n_colors=df['insurance_label'].nunique())), s=50, alpha=0.7)
    plt.title("Insurance Companies Clusters - t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(label="Insurance Label")
    plt.savefig("visualizations/tsne_clusters.png", bbox_inches="tight")
    plt.show()
    print(" Visualization saved to visualizations/tsne_clusters.png")

if __name__ == "__main__":
    create_enhanced_visualizations()  # Executam functia pentru a crea vizualizarile îmbunatatite
