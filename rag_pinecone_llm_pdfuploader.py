import os
import sys
import time
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
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

# Initialize the embedding model and the generative model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
print("✅ Embedding model loaded.")
# Use the Gemini Pro model for text generation
llm = genai.GenerativeModel("gemini-1.5-flash")
print("✅ Generative model (gemini-1.5-flash) loaded.")

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
    index_name = "rag-app-index"
    state['index_name'] = index_name # Store the static index name in the state
    
    # Check if the index already exists before creating it
    if pc.has_index(index_name):
        pc.delete_index(index_name)
        print(f"⚠️ Existing index '{index_name}' deleted. Waiting for it to be removed...")
        # Wait until the index is no longer in the list to avoid a conflict
    
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
        
        vectors.append({
            "id": f"doc-{i}", # Use a simple ID for this example
            "values": embedding,
            "metadata": {"text": chunk_text}
        })
    index.upsert(vectors=vectors)
    print(f"✅ Successfully upserted {len(vectors)} embeddings.")

    stats = index.describe_index_stats()
    print("--- Index Statistics ---")
    print(stats)
    
    return f"✅ Document processed! Ready to answer questions from the uploaded PDF."

def generate_response(query_text, state):
    """
    Performs the semantic search on the existing Pinecone index and
    uses the results to generate a final answer with a language model.
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

    # --- 5. Generate a response using the retrieved chunks ---
    # Combine the retrieved text into a single context string
    retrieved_context = "\n---\n".join([match['metadata'].get('text', '') for match in results['matches']])
    
    # Construct a prompt for the language model
    prompt = f"""
    You are a helpful assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {retrieved_context}
    
    Question: {query_text}
    
    Answer:
    """

    # Generate the response
    try:
        response = llm.generate_content(prompt)
        # Assuming the first candidate's first part is the text content
        generated_text = response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"🔴 Generation Error: {e}"

    # Prepare the final output string, including the answer and the source chunks
    final_output = f"**Answer:**\n{generated_text}\n\n"
    #final_output += "**Sources (Retrieved Chunks):**\n"
    #for i, match in enumerate(results['matches']):
     #   score = match['score']
      #  text_content = match['metadata'].get('text', 'No text content available.')
       # final_output += f"--- Source {i+1} (Score: {score:.4f}) ---\n"
        #final_output += f"{text_content}\n\n"

    return final_output


# --- 6. Gradio Interface (using Blocks) ---
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
        fn=generate_response,
        inputs=[query_text, state],
        outputs=results_output,
        api_name="generate_response"
    )

if __name__ == "__main__":
    demo.launch()
