from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import os
from dotenv import load_dotenv
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import StorageContext

load_dotenv()
# Configure api_key
GEMINI_API_KEY= os.getenv("GEMINI_API_KEY")

# Load the HuggingFace embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Gemini(api_key=GEMINI_API_KEY, model_name="models/gemini-pro")

# Load documents
documents = SimpleDirectoryReader("./data").load_data()
print(f"Number of documents loaded: {len(documents)}")

# Create a client and a new collection
client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = client.get_or_create_collection("quickstart")

# Create a vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Generate embeddings for each document
document_embeddings = [embed_model.get_text_embedding(doc.get_text()) for doc in documents]

# Print embeddings for all documents
for i, embedding in enumerate(document_embeddings):
    print(f"Embedding for document {i}: {embedding[:5]} (Length: {len(embedding)})")

# Create the index from the document embeddings
index = VectorStoreIndex.from_documents(documents,storage_context=storage_context, show_progress=True)
print(f"Index created of type: {type(index)}")

load_client = chromadb.PersistentClient(path="./chroma_db")

# Fetch the collection
chroma_collection = load_client.get_collection("quickstart")

# Fetch the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Get the index from the vector store
index = VectorStoreIndex.from_vector_store(
    vector_store
)

# Initialize the query engine (assuming 'index' is already created and available)
test_query_engine = index.as_query_engine()
response = test_query_engine.query(input('Ask your query: '))
print(response)