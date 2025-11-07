# app.py
import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f0f0f0;
        margin-right: 2rem;
    }
    .message-content {
        font-size: 1rem;
        line-height: 1.5;
    }
    .message-time {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
    }
    .session-active {
        background-color: #e6f3ff;
        border-left: 4px solid #1f77b4;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
    }
    .session-inactive {
        background-color: #f8f9fa;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        cursor: pointer;
    }
    .session-inactive:hover {
        background-color: #e9ecef;
    }
    .document-list {
        background-color: #f8f9fa;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 3px solid #28a745;
    }
    .token-counter {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: #f8f9fa;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        color: #666;
        border: 1px solid #e9ecef;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

class SimpleSessionRAGChatbot:
    def __init__(self):
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        # Sessions management
        if 'all_sessions' not in st.session_state:
            st.session_state.all_sessions = {
                "session_1": {
                    "messages": [],
                    "documents": [],  # List to store multiple documents
                    "vector_store": None,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_tokens": 0  # Add token counter for each session
                }
            }
        
        if 'current_session' not in st.session_state:
            st.session_state.current_session = "session_1"
    
    def get_current_session_data(self):
        """Get current session data"""
        return st.session_state.all_sessions[st.session_state.current_session]
    
    def switch_session(self, session_id):
        """Switch to a different session"""
        if session_id in st.session_state.all_sessions:
            st.session_state.current_session = session_id
            st.success(f"🔄 Switched to {session_id}")
    
    def create_new_session(self):
        """Create a new session"""
        new_id = f"session_{len(st.session_state.all_sessions) + 1}"
        st.session_state.all_sessions[new_id] = {
            "messages": [],
            "documents": [],
            "vector_store": None,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tokens": 0
        }
        return new_id
    
    def delete_session(self, session_id):
        """Delete a session - FIXED: Can delete current session"""
        if len(st.session_state.all_sessions) > 1 and session_id in st.session_state.all_sessions:
            del st.session_state.all_sessions[session_id]
            # Switch to another session if we deleted the current one
            if st.session_state.current_session == session_id:
                remaining_sessions = list(st.session_state.all_sessions.keys())
                self.switch_session(remaining_sessions[0])
            st.success(f"🗑️ Session {session_id} deleted!")
        elif len(st.session_state.all_sessions) == 1:
            st.error("❌ Cannot delete the last session")
    
    def delete_current_session(self):
        """Delete the current session"""
        self.delete_session(st.session_state.current_session)
    
    def reset_current_session(self):
        """Reset current session but keep the session itself"""
        session_data = self.get_current_session_data()
        session_data["messages"] = []
        session_data["documents"] = []
        session_data["vector_store"] = None
        session_data["total_tokens"] = 0  # Reset token count
        st.success("🔄 Session reset!")
    
    def check_imports(self):
        """Check if all required imports are available"""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_community.vectorstores import Chroma
            return True
        except ImportError as e:
            st.error(f"❌ Missing import: {e}")
            return False
    
    def initialize_llm(self):
        """Initialize LLM and embedding model"""
        try:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("❌ OPENAI_API_KEY not found")
                return False, None, None
            
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
            
            return True, llm, embedding_model
        except Exception as e:
            st.error(f"❌ Error initializing LLM: {str(e)}")
            return False, None, None
    
    def estimate_tokens(self, text):
        """Simple token estimation (rough approximation)"""
        # Rough estimation: ~4 characters per token for English text
        if not text:
            return 0
        return len(text) // 4
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded PDF file and add to current session documents"""
        try:
            # Check imports
            if not self.check_imports():
                return False
            
            # Initialize LLM
            success, llm, embedding_model = self.initialize_llm()
            if not success:
                return False
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Import document loader
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma
            
            # Load the PDF document
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            if not docs:
                st.error("❌ No content found in PDF")
                return False
            
            # Extract content for verification
            file_content = "\n".join([doc.page_content for doc in docs])
            
            st.info(f"📄 Loaded {len(docs)} pages from PDF")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(docs)
            
            st.info(f"✂️ Split into {len(splits)} chunks")
            
            session_data = self.get_current_session_data()
            
            # If this is the first document, create new vector store
            if session_data["vector_store"] is None:
                collection_name = f"{st.session_state.current_session}_{datetime.now().strftime('%H%M%S')}"
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embedding_model,
                    collection_name=collection_name
                )
                session_data["vector_store"] = vectorstore
            else:
                # Add to existing vector store
                existing_store = session_data["vector_store"]
                existing_store.add_documents(splits)
            
            # Add document to session's document list
            document_info = {
                "name": uploaded_file.name,
                "content": file_content[:500] + "..." if len(file_content) > 500 else file_content,
                "pages": len(docs),
                "chunks": len(splits),
                "uploaded_at": datetime.now().strftime("%H:%M:%S")
            }
            session_data["documents"].append(document_info)
            
            # Store LLM and embedding model in session
            session_data["llm"] = llm
            session_data["embedding_model"] = embedding_model
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            st.success(f"✅ '{uploaded_file.name}' added to session!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            return False
    
    def remove_document(self, doc_index):
        """Remove a document from current session"""
        session_data = self.get_current_session_data()
        if 0 <= doc_index < len(session_data["documents"]):
            removed_doc = session_data["documents"].pop(doc_index)
            # Reset vector store when documents are removed (simplest approach)
            session_data["vector_store"] = None
            st.success(f"🗑️ Removed document: {removed_doc['name']}")
            # Re-process remaining documents
            self.reprocess_documents()
    
    def reprocess_documents(self):
        """Re-process all documents in current session (after removal)"""
        session_data = self.get_current_session_data()
        if not session_data["documents"]:
            return
        
        # This would require storing the actual document content
        # For now, we'll just reset and inform user to re-upload
        st.info("📄 Please re-upload documents after removal")
        session_data["vector_store"] = None
    
    def get_rag_response(self, query):
        """Get response using RAG for current session - WITH CONVERSATION HISTORY"""
        try:
            session_data = self.get_current_session_data()
            
            if session_data["vector_store"] is None or not session_data["documents"]:
                return "Please upload at least one PDF document first.", 0
            
            vectorstore = session_data["vector_store"]
            llm = session_data["llm"]
            
            # Get relevant documents from ALL uploaded PDFs
            docs = vectorstore.similarity_search(query, k=3)
            
            if not docs:
                return "No relevant information found in the documents.", 0
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create conversation history from previous messages
            conversation_history = ""
            for msg in session_data["messages"][-6:]:  # Last 6 messages for context
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_history += f"{role}: {msg['content']}\n"
            
            # Create prompt with conversation history
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant. Use the following context from the documents to answer the question. 
                Also consider the conversation history to maintain context and provide coherent responses.
                
                If the information is not in the context, you can use your general knowledge to provide a helpful response.
                
                Document Context: {context}
                
                Conversation History: {history}
                
                Current Question: {input}
                
                Provide a helpful and comprehensive answer:"""),
            ])
            
            # Create chain
            chain = prompt_template | llm
            
            # Get response
            response = chain.invoke({
                "context": context,
                "history": conversation_history,
                "input": query
            })
            
            # Estimate tokens for this interaction
            prompt_tokens = self.estimate_tokens(query + context + conversation_history)
            response_tokens = self.estimate_tokens(response.content)
            total_tokens = prompt_tokens + response_tokens
            
            # Update session token count
            session_data["total_tokens"] += total_tokens
            
            return response.content, total_tokens
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            return error_msg, 0
    
    def display_chat_messages(self):
        """Display chat messages for current session - NO SOURCES"""
        session_data = self.get_current_session_data()
        
        for message in session_data["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(f'<div class="message-content">{message["content"]}</div>', unsafe_allow_html=True)
                
                if message.get("timestamp"):
                    st.markdown(f'<div class="message-time">{message["timestamp"]}</div>', unsafe_allow_html=True)
    
    def clear_current_chat(self):
        """Clear chat history for current session"""
        session_data = self.get_current_session_data()
        session_data["messages"] = []
        session_data["total_tokens"] = 0  # Reset token count
        st.success("💬 Chat history cleared!")
    
    def display_token_counter(self):
        """Display token counter in bottom right corner"""
        session_data = self.get_current_session_data()
        total_tokens = session_data.get("total_tokens", 0)
        
        # Format the token count (add commas for readability)
        formatted_tokens = f"{total_tokens:,}"
        
        st.markdown(
            f'<div class="token-counter">Tokens used: {formatted_tokens}</div>',
            unsafe_allow_html=True
        )

def main():
    # Initialize chatbot
    chatbot = SimpleSessionRAGChatbot()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Multi-Session RAG Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Key check
        if not os.getenv("OPENAI_API_KEY"):
            st.error("🔑 OPENAI_API_KEY not found")
            st.info("Add your OpenAI API key to a .env file:")
            st.code("OPENAI_API_KEY=your-api-key-here")
        
        # Check imports
        st.subheader("System Check")
        if chatbot.check_imports():
            st.success("✅ All imports working")
        else:
            st.error("❌ Some imports missing")
        
        # Session Management
        st.subheader("📂 Session Management")
        
        # Create new session
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Session", use_container_width=True):
                new_session_id = chatbot.create_new_session()
                chatbot.switch_session(new_session_id)
                st.rerun()
        with col2:
            if st.button("🗑️ Delete Current", use_container_width=True, type="secondary"):
                chatbot.delete_current_session()
                st.rerun()
        
        # Display sessions
        st.write("**Your Sessions:**")
        
        for session_id, session_data in st.session_state.all_sessions.items():
            is_current = session_id == st.session_state.current_session
            
            if is_current:
                st.markdown(f'<div class="session-active">', unsafe_allow_html=True)
                st.write(f"🎯 **{session_id}** (Current)")
            else:
                st.markdown(f'<div class="session-inactive">', unsafe_allow_html=True)
                st.write(f"📁 {session_id}")
            
            # Session info
            col1, col2 = st.columns([3, 1])
            with col1:
                if session_data["documents"]:
                    st.write(f"📚 {len(session_data['documents'])} documents")
                else:
                    st.write("📄 No documents")
                st.write(f"💬 {len(session_data['messages'])} messages")
                st.write(f"🔢 {session_data.get('total_tokens', 0):,} tokens")
            
            with col2:
                if not is_current:
                    if st.button("Switch", key=f"switch_{session_id}", use_container_width=True):
                        chatbot.switch_session(session_id)
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Current Session Management
        st.subheader("🛠️ Current Session")
        current_data = chatbot.get_current_session_data()
        
        if current_data["documents"]:
            st.success(f"📚 **Documents loaded:** {len(current_data['documents'])}")
            
            # Show document list with remove buttons
            st.write("**Loaded Documents:**")
            for i, doc in enumerate(current_data["documents"]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f'<div class="document-list">', unsafe_allow_html=True)
                    st.write(f"📄 {doc['name']}")
                    st.write(f"📖 {doc['pages']} pages, {doc['chunks']} chunks")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    if st.button("❌", key=f"remove_{i}", use_container_width=True):
                        chatbot.remove_document(i)
                        st.rerun()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset Session", use_container_width=True):
                    chatbot.reset_current_session()
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    chatbot.clear_current_chat()
                    st.rerun()
        else:
            st.info("📄 No documents uploaded yet")
        
        # File upload - MULTIPLE FILES
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            key=f"uploader_{st.session_state.current_session}",
            help="Upload multiple PDF documents for this session",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if file is already uploaded
                current_docs = [doc["name"] for doc in current_data["documents"]]
                if uploaded_file.name not in current_docs:
                    with st.spinner(f"📄 Processing {uploaded_file.name}..."):
                        if chatbot.process_uploaded_file(uploaded_file):
                            st.success(f"✅ {uploaded_file.name} added!")
                else:
                    st.info(f"📄 {uploaded_file.name} already uploaded")
        
        # Status
        st.subheader("📊 Status")
        st.write(f"🔑 API Key: {'✅ Found' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
        st.write(f"📚 Documents: {len(current_data['documents'])}")
        st.write(f"💬 Messages: {len(current_data['messages'])}")
        st.write(f"🔢 Total Tokens: {current_data.get('total_tokens', 0):,}")

    # Main chat area
    if not os.getenv("OPENAI_API_KEY"):
        st.error("## 🔑 OpenAI API Key Required\n\nPlease add OPENAI_API_KEY to your .env file")
        return
    
    current_data = chatbot.get_current_session_data()
    
    # Current session info
    st.subheader(f"💬 Session: {st.session_state.current_session}")
    
    if current_data["documents"]:
        document_names = [doc["name"] for doc in current_data["documents"]]
        st.success(f"📚 Chatting about: **{', '.join(document_names)}**")
        
        # Show documents preview
        with st.expander("📋 Documents Overview"):
            for doc in current_data["documents"]:
                st.write(f"**{doc['name']}** ({doc['pages']} pages)")
                st.text(doc["content"])
                st.write("---")
    else:
        st.info("📄 Upload PDF documents to start chatting!")
    
    # Display chat messages
    chatbot.display_chat_messages()
    
    # Chat input
    if prompt := st.chat_input(f"Ask about your documents..."):
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        current_data["messages"].append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f'<div class="message-content">{prompt}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="message-time">{user_message["timestamp"]}</div>', unsafe_allow_html=True)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response, tokens_used = chatbot.get_rag_response(prompt)
                
                # Display response
                st.markdown(f'<div class="message-content">{response}</div>', unsafe_allow_html=True)
                
                # Timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.markdown(f'<div class="message-time">{timestamp}</div>', unsafe_allow_html=True)
            
            # Add assistant message
            assistant_message = {
                "role": "assistant",
                "content": response,
                "timestamp": timestamp
            }
            current_data["messages"].append(assistant_message)
    
    # Display token counter in bottom right corner
    chatbot.display_token_counter()

if __name__ == "__main__":
    main()