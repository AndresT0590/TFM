import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from datetime import datetime
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Merlin's Floors", page_icon="./favicon.ico")

# Inicializar el estado de sesión
if 'accepted' not in st.session_state:
    st.session_state.accepted = False
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = []
if "active_conversation_index" not in st.session_state:
    st.session_state.active_conversation_index = None

# Función para iniciar una nueva conversación
def start_new_conversation():
    if st.session_state.current_conversation:
        first_message = next(
            (msg["text"] for msg in st.session_state.current_conversation if msg["sender"] == "user"),
            "Sin descripción"
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_summary = "Conversación " + timestamp
        is_duplicate = any(
            conv["messages"] == st.session_state.current_conversation
            for conv in st.session_state.conversations
        )
        if not is_duplicate:
            st.session_state.conversations.append({
                "name": conversation_summary,
                "messages": st.session_state.current_conversation
            })
    st.session_state.current_conversation = []
    st.session_state.active_conversation_index = None
    st.session_state.accepted = False
    st.rerun()

# Función para cargar una conversación guardada
def load_conversation(index):
    if st.session_state.active_conversation_index != index:
        st.session_state.current_conversation = st.session_state.conversations[index]["messages"]
        st.session_state.active_conversation_index = index
        st.rerun()

# Sidebar
with st.sidebar:
    st.title("Historial de conversaciones")
    st.markdown("---")
    
    if st.button("Nueva Conversación"):
        start_new_conversation()

    st.markdown("### Conversaciones Guardadas")
    
    if st.session_state.conversations:
        for i, conv in enumerate(st.session_state.conversations):
            if st.button(conv["name"], key=f"load_{i}"):
                load_conversation(i)
    else:
        st.markdown("No hay conversaciones guardadas.")

    with st.container():
        image = Image.open("./android-chrome-512x512.png")
        st.image(image, width=150)

# Contenido principal
st.title("Merlin's Floors")
# Mostrar mensaje de bienvenida y términos si no han sido aceptados
if st.session_state.accepted == 'terminos':
    st.markdown("""
    <div class="terms-container">
    <h1>Términos y Condiciones del Servicio</h1>
    <p>Este documento regula el uso del asistente virtual de Merlin's Floors.</p>
    <h2>Contacto</h2>
    <p>Puedes contactarnos en:</p>
    <a href="mailto:mariafernanda.hernandez741@comunidadunir.net">mariafernanda.hernandez741@comunidadunir.net</a><br>
    <a href="mailto:wendyvanessa.castillo102@comunidadunir.net">wendyvanessa.castillo102@comunidadunir.net</a><br>
    <a href="mailto:andresfelipe.tovar735@comunidadunir.net">andresfelipe.tovar735@comunidadunir.net</a>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Aceptar"):
        st.session_state.accepted = True
        st.rerun()
    if st.button("Rechazar"):
        st.session_state.accepted = False
        st.rerun()

elif not st.session_state.accepted:
    st.markdown("""
        <div class="welcome-message">
            <h2>¡Bienvenido a Merlin's Floors!</h2>
            <p>Tu asistente experto en remodelaciones. Acepta los términos y condiciones para continuar.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Aceptar Términos y Condiciones"):
        st.session_state.accepted = True
        st.rerun()
    if st.button("Leer Términos y Condiciones"):
        st.session_state.accepted = "terminos"
        st.rerun()

else:
    st.markdown("""
    <div class="suggested-questions">
        <h2>Preguntas Sugeridas</h2>
        <p>Ejemplo: ¿Cuál es el mejor suelo para mi terraza?</p>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar historial de mensajes
    for message in st.session_state.current_conversation:
        with st.chat_message(message["sender"]):
            st.markdown(message["text"], unsafe_allow_html=True)

    # Entrada del usuario
    if user_input := st.chat_input("Escribe tu consulta aquí..."):
        st.session_state.current_conversation.append({"sender": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Buscando información..."):
            try:
                # Simulación de respuesta de IA (puedes conectar a OpenAI aquí)
                ai_response = f"Respuesta generada para: {user_input}"
                st.session_state.current_conversation.append({"sender": "assistant", "text": ai_response})
            except Exception as e:
                st.error(f"Error al procesar la consulta: {e}")

        with st.chat_message("assistant"):
            st.markdown(ai_response)
