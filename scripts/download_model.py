from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"Downloading {model_name} to local Hugging Face cache...")
model = SentenceTransformer(model_name)
print("Model downloaded successfully!")
