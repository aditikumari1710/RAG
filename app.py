import streamlit as st
import gemini_test_llm
from gemini_test_llm import rag_pipeline



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [("AI", "👋 Hello! What is your query?")]

# Display chat history
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.write(message)

# Chat input
prompt = st.chat_input("Enter the Research Paper Name")

if prompt:
    # Display user message
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    
    source_name=gemini_test_llm.get_title(prompt)
    response = rag_pipeline(prompt,source_name)  # Placeholder response
    # ------------------------

    # Display AI response
    st.session_state.messages.append(("ai", response))
    with st.chat_message("ai"):
        st.write(response)


