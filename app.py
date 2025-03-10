import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI  # Corrected Import
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import SystemMessage

# Streamlit Page Configuration
st.set_page_config(page_title="🤖 Gemini AI Data Science Tutor", page_icon="📊")

st.title("🤖 Gemini AI Data Science Tutor")

# Load API Key Securely
with open("api_key.txt") as f:
    api_key = f.read().strip()

# Configure Google Gemini API
genai.configure(api_key=api_key)

# Initialize LangChain's Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key)

# Conversation Memory Setup
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# LangChain Conversation Chain with Memory
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory
)

# System Message to Restrict AI to Data Science Topics
system_instruction = SystemMessage(
    content="You are a Data Science Tutor. Answer only data science-related questions. "
            "If a question is outside data science, politely refuse to answer."
)
st.session_state.memory.chat_memory.messages.append(system_instruction)

# Display AI Greeting
if not st.session_state.memory.buffer:
    st.markdown('<div class="ai_message">Hi❕, I am your Data Science Tutor. Ask me anything about data science!</div>', unsafe_allow_html=True)

# Display Previous Chat History
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, SystemMessage):
        continue  # Skip system instructions
    role_class = "user_message" if msg.type == "human" else "ai_message"
    st.markdown(f'<div class="{role_class}">{msg.content}</div>', unsafe_allow_html=True)

# User Input Field
user_prompt = st.chat_input("Ask a data science question...")

if user_prompt:
    # Display User Message
    st.markdown(f'<div class="user_message">{user_prompt}</div>', unsafe_allow_html=True)
    
    # Get AI Response
    response = conversation.predict(input=user_prompt)
    
    # Display AI Response
    st.markdown(f'<div class="ai_message">{response}</div>', unsafe_allow_html=True)
