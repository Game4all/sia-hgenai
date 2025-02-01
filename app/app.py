import streamlit as st
from dotenv import load_dotenv

st.title("Chatbot Interface")

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to the chatbot!"}]

# Affichage des messages (ensuring the chat remains structured)
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Prédefined questions
predefined_questions = [
    "Quelles sont les risques?",
    "Quel est le danger le plus probable?",
    "Montre-moi tous les dangers qu'il y a eu dans la région ?",
    "random question 1",
]

# Ensure predefined questions stay just above the input field
button_container = st.container()
with button_container:
    cols = st.columns(len(predefined_questions))
    for col, question in zip(cols, predefined_questions):
        with col:
            if st.button(question, key=question):
                st.session_state["messages"].append({"role": "user", "content": question})
                response = f"This is a static response for: {question}"
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.rerun()

# User input at the bottom
prompt = st.chat_input("Type your message here...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    response = f"This is a static response for: {prompt}"
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.rerun()
