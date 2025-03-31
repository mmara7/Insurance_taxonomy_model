'''
import pickle
import pandas as pd

import sys
sys.path.append('C:\\Users\\mldma\\Desktop\\Veridion_new')
from core.data_processor import DataProcessor
from core.embedding_generator import EmbeddingGenerator
from tqdm import tqdm
from utils.device_check import verify_cuda

def main():
    verify_cuda()
    
   
    with open("models/insurance_model.pkl", "rb") as f:
        model = pickle.load(f)
    
   
    new_data = pd.read_csv("data/new_companies.csv")
    processor = DataProcessor()
    new_data['combined_text'] = new_data.apply(processor.combine_features, axis=1)
    
   
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_embeddings(new_data['combined_text'].tolist())
    
  
    new_data['predicted_labels'] = [
        model.predict(embed) 
        for embed in tqdm(embeddings, desc=" Classifying companies")
    ]
    new_data.to_csv("output/classified_companies.csv", index=False)
    print(" Predictions saved to classified_companies.csv")

if __name__ == "__main__":
    main()
    '''
'''
import sys
sys.path.append('C:\\Users\\mldma\\Desktop\\Veridion_new')
import pickle
import pandas as pd
import os
from core.embedding_generator import EmbeddingGenerator
from core.data_processor import DataProcessor

def predict_and_update_csv(input_csv_path: str, output_csv_path: str, model_path: str = "models/insurance_model.pkl"):
    
    
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
   
    processor = DataProcessor()
    df = pd.read_csv(input_csv_path)
    df['combined_text'] = df.apply(processor.combine_features, axis=1)
    
   
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_embeddings(df['combined_text'].tolist())
    
 
    df['insurance_labels'] = [
        ", ".join(model.predict(embed, top_k=3))  
        for embed in embeddings
    ]
    
    # Save the updated CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f" Updated CSV saved to {output_csv_path}")

if __name__ == "__main__":
    
    input_csv = "data/ml_insurance_challenge.csv"  
    output_csv = "data/ml_insurance_challenge_with_labels.csv" 
    
    predict_and_update_csv(input_csv, output_csv)
    
import sys
sys.path.append('C:\\Users\\mldma\\Desktop\\Veridion_new')
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm  
from core.embedding_generator import EmbeddingGenerator
from core.data_processor import DataProcessor

def predict_with_confidence(model, embedder, text, threshold=0.5, top_k=3):
    
    embed = embedder.generate_embeddings([text])[0]
    query = np.array([embed]).astype('float32')
    
   
    distances, indices = model.faiss_index.search(query, k=top_k)
    similarities = 1 / (1 + distances[0])
    
    
    results = []
    for i, sim in zip(indices[0], similarities):
        if sim >= threshold:
            results.append({
                "label": model.taxonomy_labels[i],
                "score": float(sim)
            })
    
    return results

def main():
  
    print(" Loading model...")
    with open("models/insurance_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load data
    print(" Loading data...")
    processor = DataProcessor()
    df = pd.read_csv("data/ml_insurance_challenge.csv")
    df['combined_text'] = df.apply(processor.combine_features, axis=1)
    
    
    embedder = EmbeddingGenerator()
    
    
    print(" Generating predictions...")
    predictions = []
    for text in tqdm(df['combined_text'], desc="Processing", unit="companies"):
        preds = predict_with_confidence(model, embedder, text)
        predictions.append(", ".join(item["label"] for item in preds) or "UNKNOWN")
    
    df['insurance_labels'] = predictions
    
   
    output_path = "data/ml_insurance_challenge_labeled.csv"
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f" Summary:")
    print(f"- Companies processed: {len(df)}")
    print(f"- Unknown classifications: {sum(df['insurance_labels'] == 'UNKNOWN')}")

if __name__ == "__main__":
    main()
    '''