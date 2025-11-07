from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os

# ✅ Load API key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


DATA_PATH = "data"          # folder containing PDFs (rename to your folder name)
PERSIST_DIR = "vector_store"  # folder where Chroma DB will be saved


def build_vectorstore():
    print("📄 Loading PDFs...")

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    print(f"✅ PDF Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectordb.persist()
    print("✅ Vector Store created & saved successfully!")


if __name__ == "__main__":
    build_vectorstore()
