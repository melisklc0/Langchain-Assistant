# Langchain-Assistant

An assistant project using LangChain, designed for document retrieval, question answering, and integration with vector databases.

## Features
- Document loading and splitting
- Vector database (ChromaDB) integration
- Retrieval-augmented generation
- Support for OpenAI, Google Generative AI, HuggingFace, and more

## Getting Started

### Installation (Local)
1. Clone the repository:
   ```sh
   git clone https://github.com/melisklc0/Langchain-Assistant.git
   cd Langchain-Assistant
   ```
2. Install `uv` (if not already installed): `uv` is used for Python package management in this project.
   ```sh
   pip install uv
   ```   
2. Create a virtual environment and install dependencies:
   ```sh
   uv venv

   # Activate the virtual environment on Windows (Powershell)
   .venv\Scripts\Activate.ps1 

   # Install dependencies from pyproject.toml using uv
   uv pip install -e .
   ```

4. Set up environment variables as needed (see `.env.example` if available).

5. Run the application:
   ```sh
   uv run main.py
   ```

## Project Structure
- `main.py`: Main application entry points
- `modules/`: Core logic (document loaders, retrievers, models, etc.)
- `vectorstore/`: Vector database files
- `docs/`: Example documents

## Configuration
- Edit `pyproject.toml` for Python dependencies
- Edit `.env` for environment variables (API keys, etc.)

## License
MIT License
