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
st.markdown(
    """
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
    background-image: url("./sidebar.jpg");
    background-size: cover; /* Ajusta la imagen para que cubra el área */
    background-repeat: no-repeat; /* Evita que la imagen se repita */
    background-position: center; /* Centra la imagen */
    background-color: rgba(83, 62, 45, 0.8); /* Color de fondo con opacidad para complementar */
    color: #FFFFFF !important; /* Color del texto */
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

        h2 {
            color: #ffd700;
            margin-top: 2rem;
            font-size: 1.5rem;
        }
        h3 {
            color: #ffd700;
            margin-top: 1.5rem;
            font-size: 1.2rem;
        }

        p, ul, ol {
            margin-bottom: 1rem;
            color: white;
        }

        .section {
            margin-bottom: 2rem;
        }

        .contact-info {
            background-color: rgba(255, 215, 0, 0.1);
            padding: 1rem;
            border-radius: 4px;
            margin-top: 2rem;
        }

        .contact-info a {
            color: #ffd700;
            text-decoration: none;
        }

        .contact-info a:hover {
            text-decoration: underline;
        }

        @media (max-width: 768px) {
            .terms-container {
                margin: 1rem;
                padding: 1rem;
            }
        }
        .terms-container div p {
            color: white
        }
        
        .stVerticalBlock {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        .st-emotion-cache-1flajlm {
            width: 80vh !important;
        }

        .st-emotion-cache-4oy321 {
            width: 40vw
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
        
        .suggested-questions h2{
            margin-top: 0.6rem;
        }
        

        .suggested-questions h2 {
            color: #ffd700;
            font-size: 0.9rem;
            margin-bottom: 4px;
            font-family: system-ui, -apple-system, sans-serif;
        }

        .questions-list {
            list-style: none;
            padding: 0;
            margin: 0;
            
        }
 


        .questions-list div {
            list-style-type: none; /* Quita la viñeta o número */
            
            margin-bottom: 2px;
            line-height: 1.1;
        }

        .question-text {
            width: 100%;
            background-color: rgba(255, 215, 0, 0.1);
            border: 1px solid rgba(255, 215, 0, 0.3);
            color: #e5e5e5;
            padding: 6px 8px;
            border-radius: 4px;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 0.75rem;
            display: block;
        }

        @media (max-width: 768px) {
            .suggested-questions {
                display: none;
            }
        }
        
        
        
        .st-emotion-cache-1p2n2i4{
            background-color: black !important;
        }
        .st-emotion-cache-128upt6{
            background-color: rgb(30, 32, 42) !important;
        }
        
        header{
            background-color: rgb(30, 32, 42) !important;
            color: color: #ffd700 !important;
        }
        
        .st-emotion-cache-s046zg{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            
        }
    
    
        .stVerticalBlock .stVerticalBlock .stVerticalBlock{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        } 
        
        .stVerticalBlock .stVerticalBlock h3 {
            text-align: center
        }
        
        .stVerticalBlock .stVerticalBlock p {
            text-align: center
        }
        
     
        
        .st-emotion-cache-n5r31u {
            width: 100% !important;
        }
        
        .st-emotion-cache-1igbibe {
            width: 100% !important;
        }
        
       .st-emotion-cache-phe2gf p {
           width: 100% !important;
       }
       
        .st-emotion-cache-phe2gf ol {
           width: 100% !important;
       }
           
       
       
       
       
        
        
    
    
    </style>
    """,
    unsafe_allow_html=True
)

def accept_terms():
    st.session_state.accepted = True
    st.session_state.current_conversation.append({
        "sender": "assistant",
        "text": "¡Gracias por aceptar los términos y condiciones! ¿En qué puedo ayudarte con tu proyecto de remodelación de suelos?"
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

# Template del prompt
prompt_template = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=(
        "Eres un experto asesor en suelos con más de 15 años de experiencia en España. Tu objetivo es guiar al cliente de manera clara y específica. "
        "\nPROCESO DE PENSAMIENTO para cada consulta:\n"
        "1. ANÁLISIS INICIAL:\n"
           "- ¿Qué tipo de espacio se está consultando? (cocina, baño, salón...)\n"
           "- ¿Qué requisitos específicos menciona el cliente?\n"
           "- ¿Hay restricciones de presupuesto o preferencias mencionadas?\n"
        "2. EVALUACIÓN DE NECESIDADES:\n"
           "- Nivel de tránsito en la zona\n"
           "- Exposición a humedad o condiciones especiales\n"
           "- Requisitos de mantenimiento según el usuario\n"
        "3. SELECCIÓN DE OPCIONES:\n"
           "- Identificar los 2-3 mejores materiales para el caso\n"
           "- Ordenarlos por relación calidad-precio\n"
           "- Considerar disponibilidad en la zona del cliente\n"
        "4. ANÁLISIS DE COSTES:\n"
           "- Coste del material por metro cuadrado\n"
           "- Coste de instalación estimado\n"
           "- Costes adicionales (preparación, materiales extra)\n"
        "5. FORMULACIÓN DE RESPUESTA:\n"
           "- Empezar con la recomendación principal\n"
           "- Justificar cada sugerencia\n"
           "- Incluir alternativas si aplica\n\n"
        "\nPautas de comunicación:\n"
        "- Explica siempre el PORQUÉ de tus recomendaciones\n"
        "- Divide los tipos de suelo por categorías claras:\n"
            "* Cerámicos (porcelánico, gres...)\n"
            "* Madera (tarima, parquet...)\n"
            "* Vinílicos y laminados\n"
            "* Piedra natural (mármol, pizarra...)\n"
            "* Cemento y hormigón\n"
        "- Para cada recomendación, menciona:\n"
            "* Ventajas específicas del material\n"
            "* Desventajas o limitaciones\n"
            "* Rango de precios actual en España\n"
            "* Mantenimiento necesario\n"
        "- Si te preguntan por precios:\n"
            "* Da siempre un rango (mínimo - máximo)\n"
            "* Especifica si incluye o no instalación\n"
            "* Menciona factores que pueden variar el precio\n\n"
        "REGLAS IMPORTANTES:\n"
        "1. Si no estás seguro de un dato específico, indícalo claramente\n"
        "2. Cuando hables de precios, especifica que son orientativos\n"
        "3. Si mencionan una zona específica de España, adapta tus recomendaciones al clima local\n"
        "4. Corrige amablemente si el cliente confunde términos técnicos\n"
        "5. NO respondas a preguntas sobre paredes, techos u otros elementos que no sean suelos\n\n"
        "Ejemplo de respuesta estructurada:\n"
        "'Analizando su consulta:\n"
        "1. Espacio: Cocina de alto tránsito\n"
        "2. Necesidades: Resistencia a manchas y durabilidad\n"
        "3. Recomendación: Porcelánico rectificado porque:\n"
        "   - Es el más resistente a manchas y humedad\n"
        "   - Tiene una durabilidad superior a 20 años\n"
        "   - Precio actual: 30-60€/m² (material)\n"
        "   - Instalación: 20-35€/m² adicionales'\n\n"
        "Si la pregunta NO es sobre suelos, responde:\n"
        "'Le ruego me disculpe, pero mi especialidad son exclusivamente los suelos. Para asegurarle la mejor asesoría, ¿tiene alguna consulta específica sobre tipos de pavimentos, instalación o presupuestos de suelos?'\n\n"
        "Historial de la conversación: {history}\n\n"
        "Información relevante del contexto: {context}\n\n"
        "Pregunta actual: {question}\n\n"
        "Respuesta:"
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
    
    
    
    
    with st.container():
        st.title("Historial de conversaciones")
        st.markdown("---")
        with st.container():
            
            with st.container():
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
        <h1>Términos y Condiciones del Servicio de Chatbot</h1>
        <div class="section">
            <h2>1. Introducción</h2>
            <h3>1.1. Objeto</h3>
            <p>El presente documento establece los términos y condiciones generales (en adelante, las "Condiciones") que regulan el acceso y uso del servicio de chatbot (en adelante, el "Servicio") proporcionado por Equipo Estudiantil del Máster Universitario en Visual Analytics and Big Data de la Universidad Unir de Colombia (en adelante, el "Proveedor"), basado en la tecnología de modelos de lenguaje grandes (LLM) e integrado con la API de OpenAI.</p>
            <h3>1.2. Aceptación</h3>
            <p>Al utilizar el Servicio, el usuario (en adelante, el "Usuario") acepta íntegramente y sin reservas estas Condiciones. Si el Usuario no está de acuerdo con alguna de estas Condiciones, deberá abstenerse de utilizar el Servicio.</p>
        </div>
        <div class="section">
            <h2>2. Descripción del Servicio</h2>
            <h3>2.1. Funcionalidades</h3>
            <p>El Servicio permite al Usuario mantener conversaciones con un chatbot impulsado por inteligencia artificial. El chatbot está diseñado para proporcionar información, responder preguntas y realizar tareas de manera automatizada.</p>
            <h3>2.2. Limitaciones</h3>
            <p>El Usuario reconoce que el Servicio es una herramienta de asistencia dentro ámbito de un proyecto educativo y que no debe ser utilizado para fines críticos o fuera de los lineamientos preestablecidos por la Universidad. El Proveedor no garantiza la precisión, exhaustividad o fiabilidad de la información proporcionada por el chatbot.</p>
        </div>
        <div class="section">
            <h2>3. Uso del Servicio</h2>
            <h3>3.1. Registro</h3>
            <p>Para utilizar el Servicio, el Usuario podrá usar el chatbot sin ninguna limitación dado que es un proyecto estudiantil y no requiere procesos de registro por uso y no será necesario el registro del Usuario para el uso del chatbot.</p>
            <h3>3.2. Uso lícito</h3>
            <p>El Usuario se compromete a utilizar el Servicio de forma lícita y conforme a las presentes Condiciones y a la legislación vigente colombiana en aras de cumplir con la legalidad y las normas de ley establecidas por leyes de derechos de autor y consecuentemente las leyes colombianas vigentes. Queda prohibido cualquier uso que pueda ser considerado ilegal, dañino, ofensivo, difamatorio, obsceno, amenazante o que infrinja los derechos de terceros.</p>
            <h3>3.3. Datos personales</h3>
            <p>El tratamiento de los datos personales del Usuario se regirá por la política de privacidad del Proveedor que la información será manejada únicamente por este equipo técnico y los entes de control interno ósea la Universidad y que el chatbot y las bases de datos será manejadas este mismo equipo técnico.</p>
        </div>
        <div class="section">
            <h2>8. Contacto</h2>
            <h3>8.1. Atención al cliente</h3>
            <div class="contact-info">
                <p>Para cualquier consulta o reclamación, el Usuario puede ponerse en contacto con el Proveedor a través de:</p>
                <p>
                    <a href="mailto:mariafernanda.hernandez741@comunidadunir.net">mariafernanda.hernandez741@comunidadunir.net</a><br>
                    <a href="mailto:wendyvanessa.castillo102@comunidadunir.net">wendyvanessa.castillo102@comunidadunir.net</a><br>
                    <a href="mailto:andresfelipe.tovar735@comunidadunir.net">andresfelipe.tovar735@comunidadunir.net</a>
                </p>
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
            <p>Tu asistente experto en hacer realidad tus sueños de remodelación. Estamos aquí para echarte una mano con todo lo relacionado con suelos, desde consejos hasta presupuestos aproximados.</p>
            <p><strong>Este proyecto es de carácter académico.</strong> La información proporcionada es orientativa y no será almacenada.</p>
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
                
                
                
                """,unsafe_allow_html=True)
    
    # Mostrar historial de mensajes
    for message in st.session_state.current_conversation:
        with st.chat_message(message["sender"]):
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
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=api_messages,
                        max_tokens=500,
                        temperature=0.7,
                    )
                    ai_response = response.choices[0].message.content.strip()
                    st.markdown(ai_response)
                    st.session_state.current_conversation.append({"sender": "assistant", "text": ai_response})
                except Exception as e:
                    st.error(f"Error al procesar tu solicitud: {e}")
