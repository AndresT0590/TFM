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

# Configuración de Streamlit
st.set_page_config(page_title="AOS - Asistente Virtual para Obras de Suelos", page_icon="🤖")

# Estilos CSS (mismo que en tu código original)
st.markdown(
    """
    <style>
    /* Fuente y configuración global */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body {
    background-color: #1A1924; /* Raisin Black */
    color: white;
    padding: 0 !important;
    height: 100% !important;
    width: 100% !important;
    font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    }

    /* Contenedor principal de Streamlit */
    [data-testid="stAppViewContainer"] {
    background-color: #1A1924 !important;
    background-image: none !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    }

    /* Estilos para el encabezado */
    header {
    background-color: #1A1924 !important;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    }

    h1 {
        color: #F0E68C; /* Beige claro para que combine con el esquema */
        text-align: center;
        font-size: 30px;
    }

    stBottom{
    position: sticky;
    background-color: #1A1924 !important;
    left: 0px;
    bottom: 0px;
    width: 100%;
    }

    .st-emotion-cache-hzygls {
    position: relative;
    bottom: 0px;
    width: 100%;
    min-width: 100%;
    background-color: #1A1924;
    display: flex;
    flex-direction: column;
    -webkit-box-align: center;
    align-items: center;
    }
    
    /* Sidebar diferenciado */
    section[data-testid="stSidebar"] {
        background-color: #533E2D !important;
        color: #FFFFFF !important;
        transition: all 0.3s ease;
        padding: 20px;
        box-shadow: -2px 0 5px rgba(0,0,0,0.1);
        position: relative;
    }

    /* Texto en el sidebar */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p {
        color: #FFFFFF !important;
        font-weight: 300;
    }

    /* Ajustes para el botón */
    .stButton button {
        background-color: #A27035 !important;
        color: #1A1924 !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 20px;
        font-size: 16px !important;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: 'Roboto', sans-serif;
        letter-spacing: 1px;
        display: block; /* Para asegurarse de que tome todo el ancho disponible */
        margin: 0 auto; /* Centra horizontalmente */
    }

    .stButton button:hover {
        background-color: #533E2D !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }

    /* Ajustar la barra de entrada */
    input[type="text"] {
    background-color: #FFFFFF !important;
    border: 2px solid #A27035 !important;
    color: #1A1924 !important;
    border-radius: 8px !important;
    padding: 12px 16px;
    font-size: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    }

    input[type="text"]:focus {
    outline: none !important;
    border-color: #533E2D !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Mejoras responsivas */
@media screen and (max-width: 768px) {
    body, h2, h3, h4, h5, h6, p, span, div {
        color: #FFFFFF !important; /* Forzamos el color blanco en todos los textos */
    }

    section[data-testid="stSidebar"] {
        width: 100% !important;
        padding: 10px;
    }
}

/* Scroll bar personalizada */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #1A1924;
}

::-webkit-scrollbar-thumb {
    background: #533E2D;
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: #A27035;
    </style>
    """,
    unsafe_allow_html=True
)

# MODIFICACIÓN PRINCIPAL: Obtener API KEY desde Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Configurar el cliente OpenAI
client = OpenAI(api_key=api_key)

# Configurar modelo de lenguaje
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=api_key
)

# MODIFICACIÓN: Ruta de archivo para Streamlit Cloud
data_path = "BD_Suelos.csv"  
df = pd.read_csv(data_path)

# Resto del código permanece igual que en tu versión original
df['Contenido'] = df.iloc[:, 1:].apply(lambda row: " ".join(row.astype(str)), axis=1)

@st.cache_resource
def create_vectorstore(dataframe):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    loader = DataFrameLoader(dataframe, page_content_column="Contenido")
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = create_vectorstore(df)
retriever = vectorstore.as_retriever()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres un asistente virtual experto en remodelación de suelos en España. "
        "Responde únicamente preguntas relacionadas con suelos, incluyendo: "
        "- Tipos de suelos ideales para diferentes climas o zonas (como climas húmedos o secos). "
        "- Materiales de construcción recomendados (cerámica, madera, vinilo, etc.). "
        "- Normativas locales en España relacionadas con remodelaciones de suelos. "
        "- Técnicas de instalación de suelos. "
        "- Costos aproximados de instalación y materiales en España. "
        "Ejemplos de preguntas relevantes: "
        "1. ¿Qué tipo de suelo es mejor para un baño en España? "
        "2. ¿Cuánto cuesta instalar suelos de vinilo en Madrid? "
        "3. ¿Qué normativa debo seguir para remodelar suelos en un edificio histórico en España? "
        "Si la pregunta no está relacionada con estos temas, responde estrictamente: "
        "'Lo siento, mi especialidad son los suelos y remodelaciones en España. Por favor, hazme una pregunta relacionada con este tema.' "
        "Nunca respondas preguntas filosóficas, personales, de tecnología no relacionada o temas fuera de contexto.\n\n"
        "Información relevante: {context}\n\n"
        "Pregunta: {question}\n\n"
        "Respuesta:"
    ),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt_template,
        "document_variable_name": "context",
    },
)

# Inicialización de estados de sesión
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = []
if "active_conversation_index" not in st.session_state:
    st.session_state.active_conversation_index = None

# Todas las funciones siguientes permanecen igual
def summarize_message(message):
    keywords = ["remodelación", "suelo", "habitaciones", "ayuda", "proyecto", "consulta", "remodelar"]
    words = message.split()
    filtered_keywords = [word for word in words if word.lower() in keywords]
    return " ".join(filtered_keywords).capitalize() if filtered_keywords else "Consulta general"

def start_new_conversation():
    if st.session_state.current_conversation:
        first_message = next(
            (msg["text"] for msg in st.session_state.current_conversation if msg["sender"] == "user"),
            "Sin descripción"
        )
        conversation_summary = summarize_message(first_message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_duplicate = any(
            conv["messages"] == st.session_state.current_conversation
            for conv in st.session_state.conversations
        )
        if not is_duplicate:
            st.session_state.conversations.append({
                "name": f"{conversation_summary} - {timestamp}",
                "messages": st.session_state.current_conversation
            })
    st.session_state.current_conversation = []
    st.session_state.active_conversation_index = None
    st.rerun()

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
        
for _ in range(6):
    st.sidebar.write("")

st.sidebar.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
# MODIFICACIÓN: Asegúrate de subir también el Logo.png a tu repositorio
image = Image.open("Logo.png") 
st.sidebar.image(image, width=280)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Título principal
st.title("AOS - Asistente Virtual para Obras de Suelos")

if not st.session_state.current_conversation:
    st.markdown(
        "Bienvenido a AOS, tu asistente especializado en remodelación de suelos en España. "
        "Hazme una consulta relacionada con remodelaciones de suelos para comenzar."
    )

# Mostrar historial de mensajes
for message in st.session_state.current_conversation:
    if message["sender"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["text"])
    else:
        with st.chat_message("user"):
            st.markdown(message["text"])

# Entrada del usuario
if user_input := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.current_conversation.append({"sender": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.spinner("Buscando información relevante..."):
        try:
            retrieved_docs = retriever.invoke(user_input)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            context = ""
            st.error(f"Hubo un error al buscar información relevante: {e}")
 
    formatted_prompt = prompt_template.format(context=context, question=user_input)
 
    api_messages = [{"role": "system", "content": formatted_prompt}]
    for msg in st.session_state.current_conversation:
        role = "assistant" if msg["sender"] == "assistant" else "user"
        api_messages.append({"role": role, "content": msg["text"]})
 
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=api_messages,
                    max_tokens=500,
                    temperature=0.7,
                )
                ai_response = response.choices[0].message.content.strip()
            except Exception as e:
                ai_response = f"Hubo un error al procesar tu solicitud: {e}"
            st.markdown(ai_response)
            st.session_state.current_conversation.append({"sender": "assistant", "text": ai_response})
