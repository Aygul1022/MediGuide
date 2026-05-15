# MediGuide
MediGuide is an AI-driven knowledge management agent designed for the MediSync Health Network. Developed as part of the Knowledge Management in the Digital Age curriculum during my Erasmus+ exchange at Roma Tre University, this project addresses the "Data Abundance vs. Knowledge Fragmentation" paradox in healthcare settings. The system transforms scattered clinical data into actionable knowledge, providing a Human-AI Symbiosis where the agent assists the physician's clinical wisdom rather than replacing it

This project implements a Retrieval-Augmented Generation (RAG) architecture to provide reliable, context-aware decision support.  

Brain: Powered by Phi-3-mini (via Azure AI Foundry / VS Code AI Toolkit) for local, secure, and efficient inference.


Knowledge Base: Utilizes a processed dataset of 33,955 clinical cases derived from the Medical Meadow dataset.  

Vector Store: Implemented using ChromaDB with HuggingFace Embeddings (all-MiniLM-L6-v2) for semantic search capabilities.

Interface: A professional, interactive web dashboard built with Streamlit.
