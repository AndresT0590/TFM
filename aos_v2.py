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

# Estilos CSS personalizados
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body {
        background-color: #1A1924;
        color: white;
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #1A1924 !important;
    }

    h1 {
        color: #F0E68C;
        text-align: center;
        font-size: 30px;
    }

    section[data-testid="stSidebar"] {
        background-color: #533E2D !important;
        color: #FFFFFF !important;
    }

    .stButton button {
        background-color: #A27035 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 20px;
        font-size: 16px !important;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #533E2D !important;
    }

    .welcome-message {
        background-color: #533E2D;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .welcome-message h2 {
        color: #F0E68C;
        margin-bottom: 15px;
    }

    .welcome-message p {
        color: white;
        margin-bottom: 10px;
    }

    .terms-container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: rgba(89, 67, 48, 0.9);
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .materials-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background-color: rgba(89, 67, 48, 0.9);
        border-radius: 8px;
        overflow: hidden;
    }

    .materials-table th {
        background-color: #533E2D;
        color: #F0E68C;
        padding: 12px;
        text-align: left;
        font-weight: bold;
    }

    .materials-table td {
        padding: 12px;
        border-bottom: 1px solid rgba(255, 215, 0, 0.1);
        color: white;
    }

    .materials-table tr:hover {
        background-color: rgba(255, 215, 0, 0.1);
    }

    .table-container {
        margin: 20px 0;
        overflow-x: auto;
    }

    .suggested-questions {
        position: fixed;
        right: 10px;
        top: 46%;
        transform: translateY(-50%);
        width: 180px;
        background-color: rgba(89, 67, 48, 0.95);
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        margin-right: 6px;
    }
    
    .questions-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .question-text {
        width: 100%;
        background-color: rgba(255, 215, 0, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #e5e5e5;
        padding: 6px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)

def process_table_response(response_text):
    """Procesa la respuesta para convertir las tablas de markdown a HTML"""
    if "```" not in response_text:
        return response_text

    parts = response_text.split("```")
    result = parts[0]  # Texto antes de la tabla

    for i in range(1, len(parts), 2):
        if i < len(parts):
            table_content = parts[i].strip()
            if "|" in table_content and "Material" in table_content:
                # Convertir la tabla Markdown a HTML
                html_table = '<div class="table-container"><table class="materials-table">'
                
                # Procesar las filas
                rows = [row.strip() for row in table_content.split('\n') if row.strip() and '|' in row]
                
                # Procesar encabezados
                if rows:
                    headers = [header.strip() for header in rows[0].split('|')[1:-1]]
                    html_table += '<thead><tr>'
                    for header in headers:
                        html_table += f'<th>{header.strip()}</th>'
                    html_table += '</tr></thead>'
                
                # Procesar datos (saltar la primera fila de encabezados y la segunda de separadores)
                if len(rows) > 2:
                    html_table += '<tbody>'
                    for row in rows[2:]:
                        cells = [cell.strip() for cell in row.split('|')[1:-1]]
                        if len(cells) == len(headers):  # Asegurarse de que la fila tenga el número correcto de celdas
                            html_table += '<tr>'
                            for cell in cells:
                                html_table += f'<td>{cell}</td>'
                            html_table += '</tr>'
                    html_table += '</tbody>'
                
                html_table += '</table></div>'
                result += html_table
            else:
                result += parts[i]
            
            if i + 1 < len(parts):
                result += parts[i + 1]

    return result

def generate_ai_response(user_input, api_messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=api_messages,
            max_tokens=2000,
            temperature=0.7,
            presence_penalty=0.6,
            frequency_penalty=0.3,
            stop=None
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Procesamiento especial para asegurar que la tabla se genere
        estancias = ['cocina', 'baño', 'salon', 'dormitorio', 'habitacion', 'terraza']
        if any(estancia in user_input.lower() for estancia in estancias) and '```' not in ai_response:
            # Forzar la generación de la tabla si no está presente
            new_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Genera una tabla de materiales con este formato exacto:\n```\n| Material | Marca | Precio/m² | Características | Mantenimiento |\n| -------- | ----- | --------- | --------------- | ------------- |\n| Porcelánico | Porcelanosa | 40-60€ | Alta resistencia | Limpieza simple |\n```"},
                    {"role": "user", "content": f"Genera una tabla de materiales para {user_input}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            tabla = new_response.choices[0].message.content.strip()
            ai_response = tabla + "\n\n" + ai_response
            
        processed_response = process_table_response(ai_response)
        return processed_response
        
    except Exception as e:
        return f"Error al procesar tu solicitud: {str(e)}"

def accept_terms():
    st.session_state.accepted = True
    st.session_state.current_conversation.append({
        "sender": "assistant",
        "text": "¡Bienvenidos! Me alegro de que hayáis aceptado los términos. ¿En qué puedo ayudaros con la reforma de vuestro suelo?"
    })
    st.rerun()
    
def read_terms():
    st.session_state.accepted = "terminos"
    st.rerun()
    
def rechazar_terminos():
    st.session_state.accepted = False
    st.rerun()

api_key = st.secrets["OPENAI_API_KEY"]

# Configurar el cliente OpenAI y el modelo
client = OpenAI(api_key=api_key)
llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)

# Cargar y preparar datos
data_path = "./BD_Final.csv"  
df = pd.read_csv(data_path,delimiter='\t')
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

# Template del prompt actualizado
prompt_template = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=(
        "INSTRUCCIÓN CRÍTICA: DEBES seguir este formato EXACTO al responder:\n\n"
        "SI la pregunta menciona alguna de estas palabras: cocina, baño, salón, dormitorio, habitación, terraza\n"
        "ENTONCES tu respuesta DEBE empezar SIEMPRE con:\n\n"
        "'Entendido. Para [estancia mencionada], aquí están las opciones recomendadas:\n\n"
        "```\n"
        "| Material | Marca | Precio/m² | Características | Mantenimiento |\n"
        "| -------- | ----- | --------- | --------------- | ------------- |\n"
        "| Porcelánico | Porcelanosa | 40-60€ | Alta resistencia | Limpieza diaria |\n"
        "| Gres | Roca | 30-45€ | Antideslizante | Semanal |\n"
        "| Vinílico | Tarkett | 25-35€ | Impermeable | Mensual |\n"
        "```'\n\n"
        "Y SOLO DESPUÉS de la tabla, continuar con el análisis normal.\n\n"
        "REGLA ABSOLUTA: Si detectas una estancia, la tabla DEBE ser lo primero en tu respuesta.\n\n"
        "Tras la tabla, continúa con:\n"
        "1. Análisis de necesidades\n"
        "2. Recomendaciones específicas\n"
        "3. Presupuesto y consideraciones\n\n"
        "Contexto: {context}\n"
        "Historial: {history}\n"
        "Pregunta: {question}\n\n"
        "RECUERDA: SI SE MENCIONA UNA ESTANCIA, LA TABLA VA PRIMERO."
    )
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

# Funciones de manejo de conversaciones
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
    st.session_state.accepted = False
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
    with st.container():
        with st.container():
            if st.button("Nueva Conversación"):
                start_new_conversation()
            st.markdown("### Conversaciones Guardadas")
if st.session_state:
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
   <div class="section">
       <h2>1. Introducción</h2>
       <h3>1.1. Objetivo</h3>
       <p>Este documento establece las condiciones generales que regulan el uso del asistente virtual de Merlin's Floors (en adelante, el "Servicio"), desarrollado por nuestro equipo técnico mediante tecnología de inteligencia artificial.</p>
       <h3>1.2. Aceptación</h3>
       <p>Al usar el Servicio, aceptáis estas condiciones en su totalidad. Si no estáis de acuerdo con alguna de ellas, os rogamos que no utilicéis el Servicio.</p>
   </div>
   <div class="section">
       <h2>2. Descripción del Servicio</h2>
       <h3>2.1. Funcionalidades</h3>
       <p>El Servicio os permite mantener conversaciones con nuestro asistente virtual especializado en pavimentos y reformas. Está diseñado para asesoraros y resolver vuestras dudas sobre materiales, instalación y presupuestos.</p>
       <h3>2.2. Limitaciones</h3>
       <p>Debéis tener en cuenta que este es un servicio de asesoramiento orientativo. Las recomendaciones y presupuestos son aproximados y pueden variar según la zona y el momento.</p>
   </div>
   <div class="section">
       <h2>3. Uso del Servicio</h2>
       <h3>3.1. Acceso</h3>
       <p>Podéis utilizar el servicio libremente sin necesidad de registro previo.</p>
       <h3>3.2. Uso adecuado</h3>
       <p>Os comprometéis a usar el Servicio de forma adecuada, respetando la legalidad vigente y los derechos de terceros.</p>
       <h3>3.3. Privacidad</h3>
       <p>Vuestros datos serán tratados conforme a nuestra política de privacidad, siendo gestionados exclusivamente por nuestro equipo técnico.</p>
   </div>
   <div class="section">
       <h2>4. Contacto</h2>
       <div class="contact-info">
           <p>Para cualquier consulta, podéis contactar con nosotros en:</p>
                    <a href="mailto:mariafernanda.hernandez741@comunidadunir.net">mariafernanda.hernandez741@comunidadunir.net</a><br>
                    <a href="mailto:wendyvanessa.castillo102@comunidadunir.net">wendyvanessa.castillo102@comunidadunir.net</a><br>
                    <a href="mailto:andresfelipe.tovar735@comunidadunir.net">andresfelipe.tovar735@comunidadunir.net</a>
       </div>
   </div>
</div>
    """, unsafe_allow_html=True)
    with st.container():
        if st.button("Aceptar"):
            accept_terms()
        if st.button("Rechazar"):
            rechazar_terminos()
             
elif not st.session_state.accepted:
    st.markdown("""
        <div class="welcome-message">
            <h2>¡Bienvenido a Merlin's Floors!</h2>
            <p>Tu asistente experto en hacer realidad tus sueños de remodelación. Estamos aquí para ayudarte con todo lo relacionado con suelos, desde consejos hasta presupuestos aproximados.</p>
            <p><strong>Este proyecto es de carácter académico. La información proporcionada es orientativa, por lo que podéis estar tranquilos, no será almacenada.</strong> </p>
            <p>Para continuar, por favor acepta nuestros términos y condiciones.</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        if st.button("Aceptar Términos y Condiciones"):
            accept_terms()
        
        with st.container():
            if st.button("Leer Términos y Condiciones"):
                read_terms()
    
else:
    st.markdown("""
    <div class="suggested-questions">
        <h2>Preguntas Sugeridas</h2>
        <div class="questions-list">
            <div>
                <p class="question-text">¿Cuáles son las mejores opciones de suelos para mi jardín o terraza?</p>
            </div>
            <div>
                <p class="question-text">¿Cuál es el mejor tipo de suelo para mi habitación?</p>
            </div>
            <div>
                <p class="question-text">¿Dame recomendaciones para elegir el mejor tipo de suelo para mi baño?</p>
            </div>
            <div>
                <p class="question-text">¿Qué materiales necesito para la instalación de mi suelo?</p>
            </div>
            <div>
                <p class="question-text">¿Cuál es el costo aproximado para la remodelación del suelo de mi habitación que mide 10 metros cuadrados?</p>
            </div>
        </div>
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

        with st.spinner("Buscando información relevante..."):
            try:
                retrieved_docs = retriever.invoke(user_input)
                context = "\n".join([doc.page_content for doc in retrieved_docs])
            except Exception as e:
                context = ""
                st.error(f"Error al buscar información: {e}")

        history = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in st.session_state.current_conversation[:-1]])
        
        formatted_prompt = prompt_template.format(
            context=context,
            history=history,
            question=user_input
        )

        api_messages = [{"role": "system", "content": formatted_prompt}]
        for msg in st.session_state.current_conversation:
            role = "assistant" if msg["sender"] == "assistant" else "user"
            api_messages.append({"role": role, "content": msg["text"]})

        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                try:
                    processed_response = generate_ai_response(user_input, api_messages)
                    st.markdown(processed_response, unsafe_allow_html=True)
                    st.session_state.current_conversation.append({
                        "sender": "assistant",
                        "text": processed_response
                    })
                except Exception as e:
                    st.error(f"Error al procesar tu solicitud: {e}")

