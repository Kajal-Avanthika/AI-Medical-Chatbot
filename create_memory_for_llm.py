import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# step1: Load raw pdf
DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob = '*.pdf', loader_cls = PyPDFLoader) # uses all the pdf files inside given directory
    documents = loader.load()
    return documents

documents = load_pdf_files(data = DATA_PATH)
print("Length of pdf file:", len(documents))


# step2: Create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50) # chunk overlap helps retian context

    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data = documents)
print("Length of text chunks:", len(text_chunks))

# step3: Create vector embeddings
def get_embedding_model():
    embedding_model  = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model 

embedding_model = get_embedding_model()

# step4: store embeddings in FAISS(Facebook AI Similarity Search) - stores locally in CPU
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
