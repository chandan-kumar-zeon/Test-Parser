from llama_parse import LlamaParse
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import VectorStoreIndex, Document
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os
# !sudo apt-get install poppler-utils

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llama_key = os.getenv('LLAMA_INDEX_API_KEY')

if not openai_key or not llama_key:
    raise ValueError("API keys for OpenAI or LlamaParse are not set in the environment variables.")

Settings.embed_model = OpenAIEmbedding(api_key=openai_key, model='text-embedding-ada-002')
Settings.llm = OpenAI(api_key=openai_key, model='gpt-4o-mini')

def process_docs(doc_path):
    """
    Process the uploaded PDF document using LlamaParse and PaddleOCR.

    Args:
        doc_path (str): Path to the uploaded PDF document.

    Returns:
        query_engine_llama, query_engine_paddle, images: Query engines for LlamaParse and PaddleOCR, and a list of extracted images.
    """
    parser = LlamaParse(
        api_key=llama_key,
        result_type='markdown',
        verbose=True,
        language='en',
        num_workers=2
    )
    documents = parser.load_data(doc_path)
    markdown_parser = MarkdownElementNodeParser(llm=Settings.llm, num_workers=8)
    nodes = markdown_parser.get_nodes_from_documents(documents=documents)
    base_nodes, objects = markdown_parser.get_nodes_and_objects(nodes)
    recursive_index = VectorStoreIndex(embed_model=Settings.embed_model, nodes=base_nodes + objects)
    query_engine_llama = recursive_index.as_query_engine(similarity_top_k=10)
    
    # Extract Images
    json_objects = parser.get_json_result(doc_path)
    images_folder = "./Extracted_Images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    image_dicts = parser.get_images(json_objects, download_path=images_folder)
    images = []
    for img_dict in image_dicts:
        img_path = img_dict.get('path')
        if img_path:
            img = Image.open(img_path)
            images.append(img)

    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    images_pdf = convert_from_path(doc_path, poppler_path=r"./poppler-24.02.0/Library/bin")
    documents2 = []
    for image in images_pdf:
        result = ocr.ocr(np.array(image), cls=True)
        text = "\n".join([line[1][0] for line in result[0]])
        documents2.append(text)
    documents2 = [Document(text=text) for text in documents2]
    nodes2 = markdown_parser.get_nodes_from_documents(documents=documents2)
    base_nodes2, objects2 = markdown_parser.get_nodes_and_objects(nodes2)
    recursive_index2 = VectorStoreIndex(embed_model=Settings.embed_model, nodes=base_nodes2 + objects2)
    query_engine_paddle = recursive_index2.as_query_engine(similarity_top_k=10)

    return query_engine_llama, query_engine_paddle, images

def generate_response(query, query_engine_llama, query_engine_paddle):
    """
    Generate responses from the query engines based on the input query.

    Args:
        query (str): User query.
        query_engine_llama: LlamaParse query engine.
        query_engine_paddle: PaddleOCR query engine.

    Returns:
        res1, res2: Responses from LlamaParse and PaddleOCR engines.
    """
    res1 = query_engine_llama.query(query).response
    res2 = query_engine_paddle.query(query).response
    return res1, res2
