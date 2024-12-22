import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set up device for embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and process PDF
loader = PyPDFLoader(file_path=r"C:\Users\PC\Downloads\Chat_with_PDF-main\Chat_with_PDF-main\Sachin.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data)

# Initialize LLM
llm_answer_gen = LlamaCpp(
    streaming=True,
    model_path=r"C:\Users\PC\Downloads\mistral-7b-openorca.Q4_0 (1).gguf",
    temperature=0.75,
    top_p=1,
    f16_kv=True,
    verbose=False,
    n_ctx=4096
)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

# Initialize vector store
vector_store = Chroma.from_documents(text_chunks, embeddings, persist_directory="./db")

# Set up retrieval chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
answer_gen_chain = ConversationalRetrievalChain.from_llm(llm=llm_answer_gen, retriever=vector_store.as_retriever(),
                                                         memory=memory)

# Interactive QA
while True:
    user_input = input("Enter a question (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    answers = answer_gen_chain.run(user_input)
    print("Answer:", answers)
