from pathlib import Path

import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


PROJECT_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = PROJECT_DIR / "vectorstore" / "db_faiss"
LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MEDICAL_REFERENCE_MAX_DISTANCE = 1.0

CUSTOM_PROMPT_TEMPLATE = """Use only the information in the context to answer the user's question.
If the context does not contain the answer, say you don't know. Do not invent medical facts.
If the question is not about medicine or health, say you can only answer medical questions.
Answer in 2-4 short sentences and no more than 90 words.

Context:
{context}

Question: {question}

Answer:
"""


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    return FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def get_qa_chain():
    vectorstore = get_vectorstore()
    llm = HuggingFacePipeline.from_model_id(
        model_id=LOCAL_MODEL_ID,
        task="text-generation",
        device=-1,
        pipeline_kwargs={
            "max_new_tokens": 160,
            "temperature": 0.4,
            "do_sample": True,
            "return_full_text": False,
        },
    )
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def is_medical_reference_question(vectorstore, question):
    matches = vectorstore.similarity_search_with_score(question, k=1)
    return bool(matches) and float(matches[0][1]) <= MEDICAL_REFERENCE_MAX_DISTANCE


def main():
    st.set_page_config(page_title="MediBot", page_icon="🩺")
    st.title("MediBot")
    st.caption("Ask questions about the medical reference documents.")
    st.caption("The model runs locally on CPU, so answers may take several seconds.")
    st.warning(
        "MediBot is for educational purposes only and is not a substitute for "
        "professional medical advice, diagnosis, or treatment."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not (DB_FAISS_PATH / "index.faiss").is_file() or not (
        DB_FAISS_PATH / "index.pkl"
    ).is_file():
        st.error(
            "The FAISS knowledge base is missing. Run "
            "`create_memory_for_llm.py` from the project folder first."
        )
        st.stop()

    with st.spinner("Loading the local model and medical knowledge base..."):
        vectorstore = get_vectorstore()
        qa_chain = get_qa_chain()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(source)

    question = st.chat_input("Ask a question")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            sources = []
            if not is_medical_reference_question(vectorstore, question):
                answer = (
                    "I can only answer medical questions supported by the "
                    "medical reference documents."
                )
            else:
                with st.spinner("Searching the reference and generating an answer..."):
                    result = qa_chain.invoke({"query": question})
                answer = result["result"].strip()

                for document in result["source_documents"]:
                    source_path = document.metadata.get("source", "Unknown source")
                    page = document.metadata.get("page")
                    page_text = f", page {page + 1}" if isinstance(page, int) else ""
                    citation = f"- `{Path(source_path).name}{page_text}`"
                    if citation not in sources:
                        sources.append(citation)

            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    st.markdown("\n".join(sources))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
