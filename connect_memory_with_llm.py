import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step1: Setup LLM  (Mistral with huggingface)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(hugging_face_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = hugging_face_repo_id,
        temperature = 0.4,
        max_new_tokens = 512,
        huggingfacehub_api_token= HF_TOKEN,
    )
    return llm


# Step2: Connect LLM with FAISS and create chain

CUSTOM_PROMPT_TEMPLATE = """ Uses the pieces of information provided in the context to answer user's question.
If you dont know the ans, just say you dont know, don't try to make up ans answer.
Don't provide anything out of given context

Context:{context}
Question:{question}

Start the answer directly,no small talk please.

"""

def set_custom_propmt(custom_prompt_template):
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = ["context","question"])
    return prompt

# Load Database

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH , embedding_model, allow_dangerous_deserialization= True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(HUGGING_FACE_REPO_ID),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {'k':3} ), # uses the top 3 semantic wise docs
    return_source_documents = True,
    chain_type_kwargs = {"prompt": set_custom_propmt(CUSTOM_PROMPT_TEMPLATE)} 
)

# Now invoke with a single query
user_query = input("Write query here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT:",response["result"])
print("SOURCE_DOCUMENTS:", response["source_documents"])
