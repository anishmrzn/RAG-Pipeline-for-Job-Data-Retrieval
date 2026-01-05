Setup & Installation

This project requires Python 3.9+ and runs locally using FastAPI.

1️⃣ Clone the Repository
git clone <your-github-repo-url>
cd <repo-name>

2️⃣ Create & Activate Virtual Environment
python -m venv .venv
# .venv\Scripts\activate    # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Environment Configuration
Create a .env file in the project root:

DATASET_PATH=data/jobs.csv
FAISS_INDEX_PATH=data/jobs.index
METADATA_PATH=data/jobs_metadata.json

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5

LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=500

GOOGLE_API_KEY=your_api_key_here

5️⃣ Build the Vector Index (One-time)
python scripts/build_index.py


This step:

Cleans and chunks job descriptions

Generates embeddings

Stores vectors in FAISS

6️⃣ Run the API
uvicorn app.main:app --reload

Access:

API → http://127.0.0.1:8000

Swagger UI → http://127.0.0.1:8000/docs

📘 Full Documentation

A detailed documentation report is available here:

👉 Google Docs:

<PASTE YOUR GOOGLE DOC LINK HERE>


The document includes:

High-level architecture & design decisions

Setup instructions (with screenshots)

API usage examples

Assumptions, limitations, and future improvements