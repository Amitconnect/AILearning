import os
import sys
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import uuid
import gradio as gr

# --- 1. Configuration and API Key Setup ---

# Set up your Google API key from environment variables
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    print("✅ Google Generative AI configured.")
except Exception as e:
    # Gradio will display this message if the application fails to start
    print(f"🔴 Application Error: {e}")
    sys.exit(1)

# Set up your Pinecone API key and environment
# Replace 'YOUR_PINEONE_API_KEY' with your actual Pinecone API key.
try:
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not found. Please provide it.")
    pc = Pinecone(api_key=pinecone_api_key)
    print("✅ Pinecone client initialized.")
except Exception as e:
    # Gradio will display this message if the application fails to start
    print(f"🔴 Pinecone Error: {e}")
    sys.exit(1)

# Initialize the embedding model outside the function for efficiency
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
print("✅ Embedding model loaded.")

# --- 2. Separate Processing and Querying Functions ---

def process_pdf(pdf_file, state):
    """
    Loads and processes the PDF, and builds the Pinecone index.
    The index name is returned and stored in Gradio's state.
    """
    if pdf_file is None:
        return "Please upload a PDF file first."

    try:
        # Load the PDF from the temporary file path provided by Gradio
        loader = PyPDFLoader(pdf_file.name)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages from uploaded PDF.")
    except Exception as e:
        return f"🔴 Document Loading Error: {e}"

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ Split document into {len(splits)} chunks.")
    # --- 3. Pinecone Indexing ---
    index_name = ("gemini-index")
    state['index_name'] = index_name # Store the index name in the state
    
    # Check for and delete existing index with the same name.
    # The pc.list_indexes().names is the correct syntax for the Pinecone v2.x client.
    if pc.has_index(index_name):
       pc.delete_index(index_name)
       print(f"⚠️ Existing index '{index_name}' deleted.")

    # Create a new Pinecone index
    pc.create_index(
        name=index_name,
        dimension=3072, # Dimension for the Gemini embedding model
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )
    index = pc.Index(index_name)
    print(f"✅ Pinecone index '{index_name}' created.")

    # Upsert the embeddings into Pinecone
    print("🚀 Upserting chunks into Pinecone...")
    vectors = []
    for i, doc in enumerate(splits):
        chunk_text = doc.page_content
        embedding = embedding_model.embed_query(chunk_text)
        vector_id = str(uuid.uuid4())

        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": chunk_text}
        })
    index.upsert(vectors=vectors)
    print(f"✅ Successfully upserted {len(vectors)} embeddings.")

    stats = index.describe_index_stats()
    print("--- Index Statistics ---")
    print(stats)
    
    return f"✅ Document processed! Ready to answer questions from the uploaded PDF."

def query_index(query_text, state):
    """
    Performs the semantic search on the existing Pinecone index.
    """
    index_name = state.get('index_name')
    if not index_name:
        return "Please process a PDF first."
    
    if not query_text.strip():
        return "Please enter a query."

    try:
        index = pc.Index(index_name)
    except Exception as e:
        return f"🔴 Pinecone Error: Could not connect to index '{index_name}'. {e}"

    # --- 4. Semantic Search ---
    query_vector = embedding_model.embed_query(query_text)

    results = index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True
    )

    if not results['matches']:
        return "No matches found."

    output_string = f"--- Results for Query: '{query_text}' ---\n\n"
    for i, match in enumerate(results['matches']):
        score = match['score']
        text_content = match['metadata'].get('text', 'No text content available.')
        output_string += f"Match {i+1}: (Score: {score:.4f})\n"
        output_string += f"Content: {text_content}\n\n"

    return output_string


# --- 5. Gradio Interface (using Blocks) ---
with gr.Blocks(title="RAG-powered Document Q&A") as demo:
    gr.Markdown(
        """
        # RAG-powered Document Q&A
        Upload a PDF file and process it to create a searchable index. Then, ask questions about its content.
        This app separates the document processing from the querying, allowing for multiple queries on the same document.
        """
    )
    
    state = gr.State({})  # Initialize state for storing the index name

    with gr.Row():
        pdf_file = gr.File(label="1. Upload PDF Document")
        process_button = gr.Button("Process PDF")
    
    status_output = gr.Textbox(label="Processing Status")
    
    with gr.Column():
        gr.Markdown("---")
        query_text = gr.Textbox(
            label="2. Enter your query here...", 
            lines=2, 
            placeholder="E.g., What are the key findings of this report?"
        )
        query_button = gr.Button("Search")
        
    results_output = gr.Textbox(label="Query Results", lines=10)

    # Define the data flow using event listeners
    process_button.click(
        fn=process_pdf,
        inputs=[pdf_file, state],
        outputs=status_output,
        # Set a long timeout for the processing step
        api_name="process_pdf"
    )
    
    query_button.click(
        fn=query_index,
        inputs=[query_text, state],
        outputs=results_output,
        api_name="query_index"
    )

if __name__ == "__main__":
    demo.launch()
