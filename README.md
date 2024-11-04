# Medical Chatbot with PDF Data Extraction

This project implements a medical chatbot that utilizes PDF document data to answer user queries about medical topics. The chatbot uses LangChain for managing language models and embeddings, and Pinecone as a vector database for storing and retrieving information.

## Features

- **PDF Data Extraction**: Loads medical information from PDF files in a specified directory.
- **Text Chunking**: Splits large text data into manageable chunks for better processing.
- **Hugging Face Embeddings**: Leverages pre-trained embeddings for improved query understanding.
- **Vector Database**: Uses Pinecone to store and retrieve text embeddings efficiently.
- **Conversational Interface**: Allows users to ask questions and receive answers based on the extracted data.

## Requirements

- Python 3.7 or higher
- Jupyter Notebook (for testing and development)
- Necessary libraries:
  - `langchain`
  - `pinecone-client`
  - `huggingface-hub`
  - `PyPDF2` (for PDF loading)

**Note:** Set up your Pinecone account and obtain an API key. Replace the PINECONE_API_KEY in the code with your actual API key.

## Usage
**1. Place your PDF files in the data/ directory. The code is set to load all PDF files from this directory.**

**2. Run the main script:**
```bash
python main.py
```
**3. Interact with the chatbot by typing your queries into the console. Type exit to quit the program.**

## Code Overview
* load_pdf(data): Loads PDF files from the specified directory and returns the extracted documents.
* text_split(extracted_data): Splits the extracted documents into smaller text chunks for better processing.
* download_hugging_face_embeddings(): Downloads the Hugging Face embeddings model for processing user queries.
* Pinecone Initialization: Initializes the Pinecone vector database and stores the embeddings for text chunks.
* RetrievalQA: Creates a retrieval-based question-answering system using the loaded embeddings and user queries.

## Acknowledgments
* LangChain for the language processing framework.
* Pinecone for the vector database solution.
* Hugging Face for the pre-trained models and embeddings

  
### Customization Tips
1. **Replace `yourusername`**: Update the GitHub URL to point to your actual repository.
2. **Dependencies**: Ensure the `requirements.txt` file is created with all necessary libraries. You can generate it using:
   ```bash
   pip freeze > requirements.txt
```
