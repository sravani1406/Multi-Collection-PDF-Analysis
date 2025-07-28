from sentence_transformers import SentenceTransformer

# Download and save model to local ./model directory
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("./model")

print("✅ Model downloaded and saved successfully to ./model/")
