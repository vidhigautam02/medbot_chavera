import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Load environment variables (use only if using .env file locally)
# load_dotenv()

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["general"].get("GOOGLE_API_KEY", None)

if not api_key:
    st.error("Google API key not found in secrets. Please configure it in Streamlit secrets.")
    st.stop()  # Stop further execution if API key is not found

# Path to the FAISS index
faiss_index_path = "faiss_index/index.faiss"

# Function to create or update the FAISS index
def initialize_index():
    """
    Initialize or update the FAISS index with your PDFs.
    Call this function once to create or update the index, then comment it out.
    """
    # Paths to your PDF files
    PDF_PATHS = [
        r"C:\Users\vidhi\Desktop\chavera medbot\Common Diseases and Conditions.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\Common Symptoms and Their Potential Diagnoses.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\First Aid Basics.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\General Health Knowledge.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\Medications and Treatment Options.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\Promoting Healthy Habits.pdf",
        r"C:\Users\vidhi\Desktop\chavera medbot\Women's Health Overview.pdf"
    ]
    
    embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/text-embedding-004")
    
    # Initialize a list to hold text chunks and their sources
    all_chunks_with_sources = []
    
    for pdf_path in PDF_PATHS:
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
                chunks = text_splitter.split_text(text)
                chunks_with_sources = [(chunk, {"source": os.path.basename(pdf_path)}) for chunk in chunks]
                all_chunks_with_sources.extend(chunks_with_sources)
        except Exception as e:
            st.error(f"Error reading {pdf_path}: {e}")

    if all_chunks_with_sources:
        text_chunks, metadata = zip(*all_chunks_with_sources)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
        vector_store.save_local(faiss_index_path)
        st.success("FAISS index created or updated successfully.")
    else:
        st.error("No valid PDF data to process. Please check your PDF files.")

def query(question, chat_history):
    """
    This function handles querying the chatbot.
    Parameters:
    - question: The user's query as a string.
    - chat_history: A list of tuples containing the history of question-answer pairs.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/text-embedding-004")
        
        # Load FAISS index
        new_db = FAISS.load_local(faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        
        # Initialize the LLM model
        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
        
        # Initialize a Conversational Retrieval Chain
        query_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=new_db.as_retriever(),
            return_source_documents=True
        )
        
        response = query_chain({"question": question, "chat_history": chat_history})
        
        # Extract source documents
        source_docs = response.get("source_documents", [])
        
        # Only attribute sources if they directly contribute to the response
        if source_docs:
            relevant_sources = []
            relevant_content = ""
            for doc in source_docs:
                if question.lower() in doc.page_content.lower():
                    relevant_sources.append(doc.metadata['source'])
                    relevant_content += doc.page_content
            
            if relevant_sources:
                answer = response.get("answer", "").strip()
                return {"answer": answer, "sources": list(set(relevant_sources))}
            else:
                return {"answer": "I'm sorry, I don't have more information about this context.", "sources": []}
        else:
            # If no relevant content is found in the PDFs
            return {"answer": "I'm sorry, I don't have more information about this context.", "sources": []}
    except Exception as e:
        st.error(f"An error occurred while querying: {str(e)}")
        return {"answer": "Sorry, I couldn't process your query due to an error.", "sources": []}

def show_ui():
    """
    Sets up the Streamlit UI for the Chavera MedBot chatbot.
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
                    
                    # Only display sources if they exist
                    if response.get('sources'):
                        st.write(f"**Source(s):** {', '.join(response['sources'])}")

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response.get('answer', 'Sorry, I couldn\'t find an answer.')})
                
                # Only store sources if they exist
                if response.get('sources'):
                    st.session_state.messages.append({"role": "assistant", "content": f"**Source(s):** {', '.join(response['sources'])}"})
                
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
    # Uncomment the line below to initialize or update the FAISS index
    #initialize_index()

    # Comment the line above after the index is created/updated
    show_ui()
