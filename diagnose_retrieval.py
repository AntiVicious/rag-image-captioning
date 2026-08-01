"""
Throwaway diagnostic: shows whether ChromaDB has any captions indexed, and
what RAGRetriever actually retrieves for a given image.

    python diagnose_retrieval.py path/to/image.jpg
"""

import sys

from src.config import Config
from src.database import DatabaseManager
from src.image_preprocessing import ImagePreprocessor
from src.models import ModelManager
from src.rag_retrieval import RAGRetriever

image_path = sys.argv[1]

config = Config()
model_manager = ModelManager(config)
db_manager = DatabaseManager(config)
preprocessor = ImagePreprocessor(config)

print("Loading CLIP...")
model_manager.load_clip_model()

print("Opening database at", config.chroma_db_dir)
db_manager.initialize()
stats = db_manager.get_stats()
print("Database stats:", stats)

if stats["total_embeddings"] == 0:
    print("\n*** Collection is EMPTY. No captions have been indexed yet. ***")
    print("Retrieval will always return an empty string until the database is built:")
    print("    python -m src.cli build-db")
    print("(needs the COCO dataset downloaded first -- see src/utils.py / README)")
    sys.exit(0)

retriever = RAGRetriever(config, model_manager, db_manager, preprocessor)

print("\n--- Basic retrieval (no crops) ---")
print(retriever.retrieve(image_path, use_advanced=False))

print("\n--- Advanced retrieval (with segmentation/detection crops) ---")
print(retriever.retrieve(image_path, use_advanced=True))
