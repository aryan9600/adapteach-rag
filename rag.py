import os
from PIL import Image
import torch
import google.generativeai as genai
from slugify import slugify
from dotenv import load_dotenv
from utils import DOC_STORE, load_embeddings_from_disk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_TOP_K = int(os.getenv("default_top_k", 2))

genai.configure(api_key=GOOGLE_API_KEY)
GEMINI = genai.GenerativeModel('gemini-2.5-flash')

from colpali_engine.models import ColIdefics3, ColIdefics3Processor
MODEL = ColIdefics3.from_pretrained(
    "vidore/colSmol-256M",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
PROCESSOR = ColIdefics3Processor.from_pretrained("vidore/colSmol-256M")

BASE_PROMPT = '''
You are a smart assistant designed to answer questions about textbooks available to you in the form of pdf.
You are given relevant information in the form of PDF pages. If it is not possible to answer using the provided pages,
do not attempt to provide an answer and simply say the answer is not present.
Give detailed and extensive answers, only containing info in the pages you are given.
Try to understand content and concepts using the diagrams given BUT DO NOT refer to diagrams and figures in the response.
The response should only contain text. Use markdown as the format. Queries might be generic or about a specific topic.

Query: {query}
'''

def answer_query(doc_slug: str, query: str, top_k: int = 0):
    logger.info(f"Answering query for document: {doc_slug} with query: '{query}' and top_k: {top_k}")
    try:
        image_paths, page_embeddings = load_embeddings_from_disk(doc_slug)
    except Exception as e:
        raise e
    imgs = []
    for img_path in image_paths:
        img = Image.open(img_path)
        imgs.append(img)
    
    if top_k == 0:
        top_k = DEFAULT_TOP_K
        logger.info(f"top_k not provided, using default value: {top_k}")
    if top_k == -1:
        top_k = len(imgs)
        logger.info(f"top_k is -1, using all {top_k} images")

    logger.info("Processing query to generate embeddings.")
    q_batch = PROCESSOR.process_queries([query]).to(MODEL.device)
    with torch.no_grad():
        q_embeds = MODEL(**q_batch)
    
    logger.info("Scoring and ranking images against the query.")
    scores = PROCESSOR.score_multi_vector(q_embeds, page_embeddings)[0]
    top_idxs = scores.topk(top_k).indices.tolist()
    logger.info(f"Top {top_k} indices found: {top_idxs}")

    # Prepare Gemini inputs
    logger.info("Preparing inputs for the generative model.")
    prompt_parts = [imgs[i] for i in top_idxs]
    prompt_parts.append(BASE_PROMPT.format(query=query))
    
    logger.info("Generating content with the generative model.")
    gen_resp = GEMINI.generate_content(prompt_parts)

    # Build response
    page_links = [f"/docs/{doc_slug}/page/{i}.png" for i in top_idxs]
    logger.info(f"Generated page links: {page_links}")
    
    response = {
        "answer": gen_resp.text,
        "pages": page_links
    }
    logger.info("Successfully generated the response.")
    return response
