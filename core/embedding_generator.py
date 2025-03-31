from sentence_transformers import SentenceTransformer  # Folosim un model de transformatoare pentru embeddings semantici

class EmbeddingGenerator:
    def __init__(self):
        # Folosim un model adaptat domeniului pentru embeddings
        self.model = SentenceTransformer('all-mpnet-base-v2')  # Modelul "all-mpnet-base-v2" este mai bun pentru similaritatea semantica
    
    def generate_embeddings(self, texts):
        # Generam embeddings pentru textele date
        return self.model.encode(texts, batch_size=32, show_progress_bar=True)  # Procesam textele în batch-uri de 32 și araatam bara de progres
