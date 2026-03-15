from chromadb import PersistentClient
from chromadb.utils import embedding_functions

CHROMA_DIR = "chroma_healthcare"
CHROMA_COLLECTION = "healthcare_topics"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

client = PersistentClient(path=CHROMA_DIR)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)
col = client.get_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_fn,
)

print("Total docs:", col.count())
