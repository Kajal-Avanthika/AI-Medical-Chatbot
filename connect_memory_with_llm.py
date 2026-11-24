from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Step1: Setup a local LLM
LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def load_llm(repo_id):
    return HuggingFacePipeline.from_model_id(
        model_id=repo_id,
        task="text-generation",
        device=-1,
        pipeline_kwargs={
            "max_new_tokens": 256,
            "temperature": 0.4,
            "do_sample": True,
            "return_full_text": False,
        },
    )


# Step2: Prompt
CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say you don't know. Don't make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly, no small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


# Step3: Load database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# Step4: QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(LOCAL_MODEL_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# Step5: Interactive query loop
while True:
    user_query = input("Write query here (or 'exit'): ").strip()
    if user_query.lower() in {"exit", "quit"}:
        break
    if not user_query:
        continue

    response = qa_chain.invoke({"query": user_query})
    print("RESULT:", response["result"])
