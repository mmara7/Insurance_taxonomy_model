import pandas as pd
import re
import ast
import string
from typing import Tuple

class DataProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Curata și normalizeaza textul"""
        if pd.isna(text):
            return ""  # Returneaza un string gol dacă textul este NaN
        text = str(text).lower()  # Transforma textul in litere mici
        text = re.sub(f"[{string.punctuation}]", "", text)  # Indeparteaza semnele de punctuatie
        text = re.sub(r"\d+", "", text)  # Indeparteaza cifrele
        return text.strip()  # Elimina spatiile inutile

    @staticmethod
    def combine_features(row: pd.Series) -> str:
        """Combină toate caracteristicile textuale într-un singur string"""
        features = [
            row.get('description', ''),  # Descrierea companiei
            " ".join(ast.literal_eval(row['business_tags'])) if pd.notna(row.get('business_tags')) else "",  # Etichetele de afaceri
            row.get('sector', ''),  # Sectorul
            row.get('category', ''),  # Categoria
            row.get('niche', '')  # Nisa
        ]
        return " ".join(DataProcessor.clean_text(f) for f in features).strip()    # Curata si combina
    

    @staticmethod
    def load_data(companies_path: str, taxonomy_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Încărcă și preprocesează datele de intrare"""
        taxonomy_df = pd.read_excel(taxonomy_path)  # Incarca fisierul excel
        companies_df = pd.read_csv(companies_path)  # Incarca csv
        
        companies_df['combined_text'] = companies_df.apply(DataProcessor.combine_features, axis=1)  # Cream o coloană cu textul combinat
        taxonomy_df['clean_label'] = taxonomy_df['label'].apply(DataProcessor.clean_text)  # Curatam etichetele taxonomice
        
        return companies_df, taxonomy_df  # Returnam ambele dataframe-uri
