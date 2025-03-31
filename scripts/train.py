import numpy as np
import pickle
import sys
import os
import pandas as pd
sys.path.append('C:\\Users\\mldma\\Desktop\\Veridion_new')
from core.data_processor import DataProcessor
from core.embedding_generator import EmbeddingGenerator
from core.model import InsuranceModel
from utils.device_check import verify_cuda

def main():
    verify_cuda()  # Verificam disponibilitatea GPU-ului

    # Cream directoarele necesare daca nu exista
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    # Incarcam datele
    processor = DataProcessor()
    companies, taxonomy = processor.load_data(
        "data/ml_insurance_challenge.csv",  
        "data/insurance_taxonomy.xlsx"  
    )
    
    # Generam embeddings pentru taxonomie și companii
    embedder = EmbeddingGenerator()
    print("Generating taxonomy embeddings...")
    tax_embeddings = embedder.generate_embeddings(taxonomy['clean_label'].tolist())
    print("Generating company embeddings...")
    company_embeddings = embedder.generate_embeddings(companies['combined_text'].tolist())
    
    # Initializam și antrenam modelul
    model = InsuranceModel()
    model.build_index(tax_embeddings)  # Construim indexul FAISS pentru taxonomie
    model.taxonomy_labels = taxonomy['label'].tolist()  # Setam etichetele taxonomice
    
    # Salvam modelul antrenat într-un fișier .pkl
    model_path = "models/insurance_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)  # Salvam modelul pentru utilizare ulterioara
    print(f"Model saved to {model_path}")
    
    # Generam predictii pentru toate companiile
    print("Generating predictions...")
    companies['insurance_label'] = [
        model.predict(embed, top_k=3)  # Obtinem top 3 etichete cele mai similare pentru fiecare companie
        for embed in company_embeddings
    ]
    
    # Salvam rezultatele în fisierul CSV
    output_path = "data/output/companies_with_insurance_labels.csv"
    companies.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()  # Executam functia principala pentru antrenarea modelului
