from pdf2image import convert_from_path
import torch
from slugify import slugify
import logging
from pathlib import Path
import pickle
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory store: slug -> {"images": [PIL images...], "embeddings": tensor}
DOC_STORE = {}

def save_and_process_pdf(uploaded_file_path: str, doc_name: str, processor, model) -> str:
    logger.info(f"Processing PDF: {uploaded_file_path} for document name: {doc_name}")
    slug = slugify(doc_name)
    logger.info(f"Generated slug: {slug}")
    
    logger.info(f"Converting PDF to images: {uploaded_file_path}")
    images = convert_from_path(uploaded_file_path)
    os.makedirs("images", exist_ok=True)
    os.makedirs(f"images/{slug}")
    img_paths = []
    for i, img in enumerate(images):
        img_path = f"images/{slug}/{i}.jpg"
        img.save(img_path)
        img_paths.append(img_path)

    logger.info(f"Successfully converted PDF to {len(images)} images.")
    
    logger.info("Processing images to generate embeddings.")
    batch = processor.process_images(images).to(model.device)
    with torch.no_grad():
        embeds = model(**batch)
    logger.info("Successfully generated embeddings for the images.")
    
    save_embeddings_to_disk(slug, img_paths, embeds)
    logger.info(f"Stored document in DOC_STORE with slug: {slug}")
    del embeds, batch, images
    gc.collect()
    torch.cuda.empty_cache()
    
    return slug

EMBEDDING_DIR = Path("embedding_store")
EMBEDDING_DIR.mkdir(exist_ok=True)

def get_pickle_path(slug: str) -> Path:
    return EMBEDDING_DIR / f"{slug}.pkl"

def save_embeddings_to_disk(slug, image_paths, embeddings):
    path = get_pickle_path(slug)
    with open(path, "wb") as f:
        pickle.dump((image_paths, embeddings), f)

def load_embeddings_from_disk(slug):
    path = get_pickle_path(slug)
    if not path.exists():
        raise FileNotFoundError(f"No cached embeddings for document: {slug}")
    with open(path, "rb") as f:
        return pickle.load(f)
