import os
import logging
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import requests
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Check if required environment variables are set
if not os.getenv('GOOGLE_API_KEY'):
    raise ValueError("GOOGLE_API_KEY environment variable not set")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    logging.warning(f"Text extraction failed for page in {pdf_path}.")
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
    return text

def process_pdf(pdf_path):
    """
    Processes a single PDF file: extracts text, splits it into chunks,
    and converts chunks into embeddings.
    """
    logging.info(f"Processing file: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    chunks_with_sources = [(chunk, {"source": os.path.basename(pdf_path)}) for chunk in chunks]
    return chunks_with_sources

def upload_pdfs(pdf_paths):
    """
    Processes a list of PDF files: extracts text, splits it into chunks,
    converts chunks into embeddings, and saves them into a FAISS vector store.
    """
    all_chunks_with_sources = []
    for pdf_path in pdf_paths:
        chunks_with_sources = process_pdf(pdf_path)
        all_chunks_with_sources.extend(chunks_with_sources)

    if all_chunks_with_sources:
        text_chunks, metadata = zip(*all_chunks_with_sources)
        embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv('GOOGLE_API_KEY'), model="models/text-embedding-004")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
        vector_store.save_local("faiss_index")
        logging.info("FAISS index created or updated successfully.")
    else:
        logging.warning("No valid PDF data to process. Please check your PDF files.")


def reframe_with_gemini(text,question):
    # Configure the API key from the environment variable
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Set up the generation configuration
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 1200,

    }
    
    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    
    # Prepare the prompt
    prompt = f"""
You are a medical assistant chatbot designed to provide reliable health information and assist users with medical-related questions. Analyze the user's query carefully and use the provided document text to offer a direct, relevant, and empathetic response. If the query is general or not directly covered by the document text, provide a polite response and clarify that more information is needed.

User Query: {question}
Document Text: {text}

Ensure that the response:
1. Directly addresses the specific question asked by the user.
2. If the user query is not directly related to the document, politely suggest they ask something else or consult a healthcare professional.
3. Avoids introducing irrelevant information from the document or guessing answers.
4. Encourages the user to consult a healthcare provider for more personalized or complex advice.
5. Keep responses concise and relevant, avoiding overly long or irrelevant information.
"""

    # Generate the response
    try:
        response = model.generate_content(prompt)
        
        # Extract and return the content
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text
        else:
            print("No candidates found in the response.")
            return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def generate_natural_language_response(relevant_info, question):
    """
    Generates a natural language response based on the relevant information.
    If no relevant information is found, no source or source heading will be included.
    """
    if not relevant_info:
        return "Sorry, I couldn't find any relevant information."

    response = "Here's what I found based on your question:\n\n"
    
    # Aggregate and format the information
    source_info = {}
    for text, meta in relevant_info:
        source = meta.get('source', 'Unknown')
        if source not in source_info:
            source_info[source] = []
        source_info[source].append(text)

    document_based = False
    for source, texts in source_info.items():
        aggregated_text = " ".join(texts)  # Combine all texts from the same source
        summarized_text = reframe_with_gemini(aggregated_text, question)  # Reframe combined text
        if summarized_text:
            document_based = True  # Mark as document-based if relevant information is found
            response += summarized_text + "\n\n"

    # Only return response, no source heading if not document-based
    return response.strip() if document_based else "I couldn't find a specific answer. Could you please provide more details or ask a different question?"

def extract_relevant_information(question, text_chunks, metadata):
    """
    Extracts and aggregates relevant information based on the question.
    """
    relevant_info = []
    keywords = re.findall(r'\b\w+\b', question.lower())
    
    for chunk, meta in zip(text_chunks, metadata):
        chunk_lower = chunk.lower()
        if any(keyword in chunk_lower for keyword in keywords):
            relevant_info.append((chunk, meta))
    
    return relevant_info

def query(question, chat_history):
    """
    Processes a query using the conversational retrieval chain and returns a natural language response.
    """
    try:
        # Initialize embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv('GOOGLE_API_KEY'), model="models/text-embedding-004")
        vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

        # Retrieve the relevant chunks based on the question
        search_results = vector_store.similarity_search(question)
        if not search_results:
            return {"answer": "I couldn't find any relevant information.", "sources": []}

        # Extract text and metadata from search results
        text_chunks = [result.page_content for result in search_results]
        metadata = [result.metadata for result in search_results]

        # Extract relevant information
        relevant_info = extract_relevant_information(question, text_chunks, metadata)

        # Generate a response using the reframed information
        formatted_answer = generate_natural_language_response(relevant_info,question) if relevant_info else "I couldn't find a specific answer. Could you please provide more details or ask a different question?"

        return {"answer": formatted_answer}
    
    except Exception as e:
        logging.error(f"Error during query: {e}")
        return {"answer": "Oops, something went wrong while processing your query. Please try again later."}

def show_ui():
    """
    Sets up the Streamlit UI for the Chavera MedBot chatbot without displaying sources.
    """
    st.title("Chavera MedBot")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOYTLrTipnZW-cmCRW0yjK96iMWENViez1lQ&s", width=300)  # Adjust width as needed
    st.write("Hello! I am Chavera MedBot, Your Personal Healthcare Assistant")

    # Initialize session state for chat history and messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter Your Healthcare Query"):
        try:
            # Process the user's query and get the response
            with st.spinner("Processing your query..."):
                response = query(question=prompt, chat_history=st.session_state.chat_history)

                # Display the user's query and bot's response
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    st.markdown(f"{response.get('answer', 'Sorry, I couldn\'t find an answer.')}")

                # Store the messages in the session state
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response.get('answer', 'Sorry, I couldn\'t find an answer.')})

                # Update chat history
                st.session_state.chat_history.append((prompt, response.get('answer', 'Sorry, I couldn\'t find an answer.')))
        except Exception as e:
            st.error(f"An error occurred while processing the query: {str(e)}")

    # Add a "Restart Chat" button below the output area and above the input area
    if st.button("Restart Chat"):
        # Reset the chat state
        st.session_state.messages = []
        st.session_state.chat_history = []




if __name__ == "__main__":
    pdf_paths = [
        r"C:\Users\vidhi\Desktop\chavera medbot\Common Diseases and Conditions.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\Common Symptoms and Their Potential Diagnoses.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\First Aid Basics.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\General Health Knowledge.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\Medications and Treatment Options.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\Promoting Healthy Habits.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\Women's Health Overview.pdf"
    ]
    
    # Upload and process the PDF files (this should be done once at the start)
    #upload_pdfs(pdf_paths)

    # Show the Streamlit UI
    show_ui()
