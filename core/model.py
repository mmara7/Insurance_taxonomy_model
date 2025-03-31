import faiss
import numpy as np
from typing import List

class InsuranceModel:
    def __init__(self):
        self.faiss_index = None  # Indexul FAISS va stoca embeddings pentru cautarea similaritatii
        self.taxonomy_labels = None  # Etichetele taxonomice vor fi folosite pentru a identifica categoriile

    def build_index(self, embeddings: np.ndarray):
        """Creaza indexul FAISS pentru cautarea similaritatii"""
        dim = embeddings.shape[1]  # Dimensiunea embeddings-ului
        self.faiss_index = faiss.IndexFlatL2(dim)  # Indexul FAISS foloseste distanța L2 (Euclideană)
        self.faiss_index.add(embeddings.astype('float32'))  # Adaugam embeddings-urile la index

    def predict(self, embedding: np.ndarray, top_k: int = 3, threshold: float = 0.7) -> List[str]:
        """Gaseste cele mai similare etichete din taxonomie folosind FAISS"""
        
        # Convertim embedding-ul la un format compatibil cu FAISS
        query = np.array([embedding]).astype('float32')
        
        # Cautam etichetele similare
        distances, indices = self.faiss_index.search(query, k=top_k)
        
        # Calculam scorurile de similaritate (0-1)
        similarities = 1 / (1 + distances[0])
        
        # Filtram etichetele care depasesc pragul și le returnam
        results = []
        for i, sim in zip(indices[0], similarities):
            if sim >= threshold:
                results.append(self.taxonomy_labels[i])
        
        # Daca nu gasim etichete care sa indeplineasca pragul, returnam top 1
        return results if results else [self.taxonomy_labels[indices[0][0]]]
