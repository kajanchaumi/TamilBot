import os
from dotenv import load_dotenv
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import re
# OCR imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from typing import List

def load_english_pdfs(path: str) -> List[Document]:
    """Loads text-based PDFs and adds 'lang' metadata."""
    print(f"Loading English PDFs from {path}...")
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["lang"] = "en" # Add language metadata
        doc.metadata["source_type"] = "pdf"
    return docs

def load_tamil_ocr_pdfs(path: str) -> List[Document]:
    """
    Loads image-based Tamil PDFs using Tesseract OCR.
    """
    print(f"Loading Tamil (OCR) PDFs from {path}...")
    docs = []
    for filename in os.listdir(path):
        if not filename.lower().endswith(".pdf"):
            continue
        
        file_path = os.path.join(path, filename)
        print(f"  Processing {filename}...")
        try:
            # Convert PDF pages to images
            images = convert_from_path(file_path)
            
            for i, image in enumerate(images):
                # Use pytesseract to extract Tamil text
                # You MUST have the 'tam' (Tamil) language pack installed for Tesseract
                text = pytesseract.image_to_string(image, lang='tam')
                
                if text.strip(): # Only add if text was found
                    metadata = {
                        "source": file_path,
                        "page": i + 1,
                        "lang": "ta", # Add language metadata
                        "source_type": "pdf_ocr"
                    }
                    docs.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"    Failed to process {filename} on page {i+1}: {e}")
            
    print(f"Loaded {len(docs)} pages from Tamil PDFs.")
    return docs

def load_csv_files(path: str) -> List[Document]:
    """Loads CSV files and adds 'lang' metadata."""
    print(f"Loading CSVs from {path}...")
    # --- THIS IS THE FIX ---
    # Define the arguments for CSVLoader, specifying UTF-8 encoding
    csv_loader_kwargs = {'encoding': 'utf-8'}

    loader = DirectoryLoader(
        path,
        glob="*.csv",
        loader_cls=CSVLoader,
        show_progress=True,
        loader_kwargs=csv_loader_kwargs  # Pass the encoding args to the loader
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["lang"] = "en" # Assume CSV data is English (or change as needed)
        doc.metadata["source_type"] = "csv"
    return docs

def clean_document_content(docs: List[Document]) -> List[Document]:
    """Basic cleaning for all loaded documents."""
    print("Cleaning all documents...")
    cleaned_docs = []
    for doc in docs:
        text = doc.page_content
        text = re.sub(r'\n{3,}', '\n\n', text) # Consolidate blank lines
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text) # Fix hyphenation
        text = re.sub(r'\s{2,}', ' ', text) # Consolidate whitespace
        
        if text.strip(): # Only keep if content remains
            doc.page_content = text
            cleaned_docs.append(doc)
    return cleaned_docs



"""
# Keep minimal metadata
def filter_to_minimal_docs(docs):
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs
"""

# REVISED FUNCTION to filter and clean metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filters documents to keep only essential metadata ('source')
    and ensures the 'source' field is never null.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        # Get the source from metadata.
        src = doc.metadata.get("source")
        
        # --- THIS IS THE FIX ---
        # If the source is None (null), replace it with a valid string.
        if src is None:
            src = "Unknown Source"
        
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split into text chunks
def text_split(extracted_data, chunk_size=300, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def split_docs(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50, # Increased overlap for better context
        length_function=len
    )
    texts_chunk = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(texts_chunk)}")
    return texts_chunk

# Multilingual embeddings (English + Tamil)
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"  # 1024 dimensions
    )
    return embeddings

