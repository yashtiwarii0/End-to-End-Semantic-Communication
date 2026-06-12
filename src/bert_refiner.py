from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def refine_text(text):

    embedding = model.encode(text)

    return {
        "text": text,
        "embedding_shape": embedding.shape
    }