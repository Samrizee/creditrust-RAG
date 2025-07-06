# 🔍 Project Summary

This project builds an AI-powered internal assistant for **CrediTrust Financial**, designed to transform unstructured customer complaint data into actionable insights. The system uses **Retrieval-Augmented Generation (RAG)** to answer natural language queries with concise, evidence-supported responses sourced from the CFPB complaints dataset.

---

## 📁 Directory Layout

├── data/ # Raw and processed complaint data
├── notebooks/ # EDA and prototyping notebooks
├── src/ # Source code for the pipeline and UI
├── vector_store/ # Serialized vector index for semantic search
├── reports/ # Evaluation summaries and documentation
├── screenshots/ # UI snapshots for reference
├── .gitignore # Git exclusions
├── README.md # Project documentation
└── requirements.txt # Python package dependencies

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Samrizee/creditrust-RAG.git
cd creditrust-rag


### 2. Set up a virtual environment
```bash
python -m venv venv
# Activate (Windows)
.\venv\Scripts\activate

 
### 3. Install required packages
```bash
pip install -r requirements.txt

