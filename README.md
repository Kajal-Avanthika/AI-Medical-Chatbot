# AI-Medical-Chatbot
# MediBot

MediBot is a local, retrieval-augmented medical reference chatbot. It uses Streamlit for the web interface, FAISS to retrieve relevant passages from PDF documents, and a small Hugging Face model to generate answers. The model runs on your computer; the app does not call a paid Hugging Face Inference Endpoint.

> **Medical disclaimer:** MediBot is an educational project, not a medical professional. Do not use it to diagnose or treat a health condition or for emergencies. Consult a qualified healthcare professional.

## Features

- Ask questions in a Streamlit chat interface.
- Retrieve relevant passages from PDF files and show source pages.
- Decline questions that do not match the medical reference material.
- Run the language model locally with no per-request inference charge.
- Keep chat messages in the current Streamlit session.

## Requirements

- Python 3.10 or newer
- Internet access for the initial package and model downloads
- Enough memory and disk space to load and cache the local models

The app is configured to run the language model on CPU. Replies can take several seconds, depending on your computer. Model weights are downloaded from Hugging Face on first use; no `HF_TOKEN` is required for these public models.

## Setup on Windows

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
```

If PowerShell prevents virtual-environment activation, run the commands using the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

## Add PDF documents

Put PDF files that you are authorized to use under `data\`. The current GitHub setup intentionally excludes PDFs; provide your own files after cloning.

Build the searchable FAISS index from the project folder:

```powershell
.\.venv\Scripts\python.exe .\create_memory_for_llm.py
```

This creates `vectorstore\db_faiss\index.faiss` and `vectorstore\db_faiss\index.pkl`. Re-run the command whenever you add or change PDFs.

## Run the Streamlit website

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\medibot.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>. Type a question into the chat box and press Enter. The first startup may take longer while the embedding and language models load. Type in another question after a response appears to continue the conversation.

By default, the website is available only on the computer running it. Do not expose it directly to the public internet without adding appropriate authentication and security controls.

## Command-line alternative

After building the FAISS index, the terminal chatbot can be run with:

```powershell
.\.venv\Scripts\python.exe .\connect_memory_with_llm.py
```

Enter `exit` or `quit` to stop.

## Project files

| File or directory | Purpose |
| --- | --- |
| `medibot.py` | Streamlit chat application |
| `create_memory_for_llm.py` | Loads PDFs, splits their text, and creates the FAISS index |
| `connect_memory_with_llm.py` | Terminal-based retrieval chatbot |
| `requirements-app.txt` | Python dependencies for the app and PDF-indexing script |
| `data/` | Place authorized PDF reference documents here |
| `vectorstore/db_faiss/` | Generated FAISS index; create locally and do not commit |

