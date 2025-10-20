
from dotenv import load_dotenv
import os
from src.helper import load_english_pdfs, load_tamil_ocr_pdfs, load_csv_files, clean_document_content, split_docs, filter_to_minimal_docs, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# ------------------ MAIN EXECUTION ------------------

if __name__ == "__main__":
    load_dotenv()

    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Load documents
    print("Loading English PDFs...")
    
    docs_en = load_english_pdfs("D:\BOT\TamilBot\data\d-en")
    docs_ta = load_tamil_ocr_pdfs("D:\BOT\TamilBot\data\d-ta")
    #docs_csv = load_csv_files(PATH_CSV)
    #all_docs = docs_en + docs_ta + docs_csv
    all_docs = docs_en + docs_ta 
    #print("Loading Tamil PDFs...")
    #tamil_pdfs = load_pdf_file(data_dir="D:/CyChat/CyChat/data/d-ta")

    #print("Loading CSV files...")
    #csv_docs = load_csv_file(file_path="D:/CyChat/CyChat/data/d-ta/csv/data-ta.csv")

    #print("Loading scanned PDF with OCR...")
    #ocr_docs = load_pdf_with_ocr(file_path="D:/CyChat/CyChat/data/d-ta/hithawathi.pdf")

    #print(f"OCR loaded {len(ocr_docs)} pages")
    #print(ocr_docs[0].page_content[:500])  # check Tamil text is extracted


    filter_data = filter_to_minimal_docs(all_docs)
    cleaned_docs = clean_document_content(filter_data)
    texts_chunk = split_docs(cleaned_docs)
    print(f"Total docs loaded: {len(all_docs)}")
    print(f"Split into {len(texts_chunk)} chunks")

    # Embeddings
    embeddings = download_hugging_face_embeddings()

    # Pinecone setup
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "chat1"

    if not pc.has_index(index_name):
        print("Creating Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


print("Uploading documents to Pinecone...")
# Upload in micro-batches to avoid 4MB error
batch_size = 5
for i in range(0, len(texts_chunk), batch_size):
    batch = texts_chunk[i:i+batch_size]
    PineconeVectorStore.from_documents(
        documents=batch,
        index_name=index_name,
        embedding=embeddings,
    )
    print(f"✅ Uploaded batch {i//batch_size + 1} ({len(batch)} docs)")

print("🎉 All documents uploaded successfully!")