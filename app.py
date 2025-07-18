from flask import Flask, render_template, request, redirect, url_for, flash
import random
import datetime
from pathlib import Path
import PyPDF2

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
import unicodedata
from langchain_community.llms import Ollama
import os
from werkzeug.utils import secure_filename

# --- 1. Configuración de la Aplicación Flask ---
app = Flask(__name__)

# Configuración de la carpeta de subidas
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Clave secreta para mensajes flash (¡CAMBIA ESTO EN PRODUCCIÓN!)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui_cambiala' 

# Extensiones de archivo permitidas para la subida de CSV
ALLOWED_EXTENSIONS = {'csv'}

# --- 2. Funciones de Ayuda ---

# Función para verificar si la extensión del archivo es permitida
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 3. Configuración de Modelos y Datos ---

# Chatbot 1: Flan-T5 Large (para PDF)
PDF_PATH = Path(r"Datos\riesgo-credito-2.pdf") # Asegúrate de que esta ruta sea correcta
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "google/flan-t5-large"

# Chistes financieros (para respuestas aleatorias)
financial_jokes = [
    "¿Por qué el banco no le prestó al fantasma? ¡Porque no tenía historial crediticio!",
    "¿Qué le dijo el riesgo de crédito al cliente? ¡No te preocupes, solo estoy evaluando tu *score*!",
    "¿Por qué el analista de crédito está soltero? Porque siempre ve demasiado riesgo en las relaciones."
]

# Función para extraer texto de PDF
def extract_text_with_pypdf2(path: str) -> str:
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            page_text = page_text.lower()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# Función para construir el vector store
def build_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "-", ":"]
    )
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_texts(chunks, embeddings)

# Función para construir el pipeline del LLM (Flan-T5)
def build_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150
    )
    return HuggingFacePipeline(pipeline=pipe)

# Función para construir la cadena de QA
def build_qa_chain(vector_store, llm):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Responde como experto colombiano en análisis financiero.
Da respuestas cortas y concretas, sé amistoso, y ten capacidad para conectar con los demás.
Usa únicamente la información del contexto.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}
    )

# Preparación de LLM y Vector Store (se ejecuta una vez al iniciar la app)
print("Cargando modelo y datos...")
try:
    context = extract_text_with_pypdf2(str(PDF_PATH))
    vector_store = build_vectorstore(context)
    llm = build_llm_pipeline()
    qa_chain = build_qa_chain(vector_store, llm)
    print("Modelo Flan-T5 y Vector Store cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo Flan-T5 o el PDF: {e}")
    qa_chain = None 

# Función para guardar la conversación (log)
def save_conversation(user_input: str, response: str, log_path: Path = Path("chat_history.txt")):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Usuario: {user_input}\n")
        f.write(f"[{timestamp}] Chatbot: {response}\n\n")

# Historial de chat en memoria (para la sesión actual)
chat_history = []

# Chatbot 2: Ollama (para datos CSV)
# Inicializar Ollama una sola vez
try:
    ollama_llm = Ollama(model="mistral", temperature=0.7)
    print("Modelo Ollama 'mistral' cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo Ollama 'mistral': {e}")
    ollama_llm = None 

# Diccionario de sinónimos simples para el chatbot de CSV
SINONIMOS = {
    "hipotecario": "vivienda", "crédito hipotecario": "vivienda", "credito hipotecario": "vivienda",
    "tarjeta": "tarjeta de crédito", "tarjeta de credito": "tarjeta de crédito",
    "carro": "vehículo", "auto": "vehículo", "libre": "libre inversión",
    "microcreditos": "microcrédito", "microcredito": "microcrédito",
    "educativo": "créditos educativos diferentes a libranza"
}

# Normalizar texto (para búsqueda en CSV)
def normalizar(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return texto.strip()

# Buscar nombre real de columna en DataFrame
def encontrar_columna(df, nombre_objetivo):
    nombre_objetivo = normalizar(nombre_objetivo)
    for col in df.columns:
        if normalizar(col) == nombre_objetivo:
            return col
    raise KeyError(f"No se encuentra la columna '{nombre_objetivo}' en el archivo CSV.")

# Filtrar productos en DataFrame basado en la pregunta
def filtrar_productos(df, pregunta):
    pregunta_norm = normalizar(pregunta)
    for k, v in SINONIMOS.items():
        if k in pregunta_norm:
            pregunta_norm = pregunta_norm.replace(k, v)

    col_producto = encontrar_columna(df, 'producto de credito')
    col_entidad = encontrar_columna(df, 'nombre_entidad')

    productos = df[col_producto].dropna().unique()
    entidades = df[col_entidad].dropna().unique()

    productos_norm = [(p, normalizar(p)) for p in productos]
    entidades_norm = [(e, normalizar(e)) for e in entidades]

    coincidencias_producto = [p for p, pn in productos_norm if all(palabra in pn for palabra in pregunta_norm.split() if len(palabra) > 3)]
    coincidencias_entidad = [e for e, en in entidades_norm if any(palabra in en for palabra in pregunta_norm.split() if len(palabra) > 3)]

    if coincidencias_entidad:
        df = df[df[col_entidad].isin(coincidencias_entidad)]

    if coincidencias_producto:
        df = df[df[col_producto].isin(coincidencias_producto)]
        return df, coincidencias_producto[0] if coincidencias_producto else None

    col_tipo = encontrar_columna(df, 'tipo_de_credito')
    tipos = df[col_tipo].dropna().unique()
    tipos_norm = [(t, normalizar(t)) for t in tipos]
    tipo_match = [t for t, tn in tipos_norm if any(palabra in tn for palabra in pregunta_norm.split())]
    if tipo_match:
        return df[df[col_tipo].isin(tipo_match)], tipo_match[0]

    return pd.DataFrame(), None

# Función principal del chatbot de datos (Ollama)
def data_chat_bot_2(pregunta: str, df: pd.DataFrame):
    if ollama_llm is None: 
        return "Lo siento, el modelo de IA para análisis de datos no está disponible en este momento."

    df_filtrado, producto_detectado = filtrar_productos(df, pregunta)
    
    if df_filtrado.empty:
        return "No encontré datos relevantes para tu consulta en el archivo CSV. ¿Podrías ser más específico?"

    col_fecha = encontrar_columna(df, 'by_month_fecha_corte')
    if col_fecha not in df_filtrado.columns:
        return f"Error: La columna de fecha '{col_fecha}' no se encontró en los datos filtrados."

    muestra = df_filtrado.sort_values(col_fecha, ascending=False).head(30).to_markdown(index=False)

    prompt = f"""
    Eres un asistente financiero experto en tasas de interés en Colombia. A continuación tienes información sobre productos financieros.
    Consulta del usuario: "{pregunta}"
    Datos relevantes:
    {muestra}
    Por favor, responde en español, de forma clara y explicativa. Si el usuario pregunta por una entidad, indica cuáles son los productos financieros y sus tasas ofrecidos por esa entidad. Si pregunta por un producto, asegúrate de que el resultado corresponda al producto indicado, y no a otro producto distinto aunque tenga una tasa más baja. Si la consulta menciona tipo de persona (natural o jurídica), tenlo en cuenta al seleccionar los resultados. Si el usuario pregunta por una entidad, indica cuáles son los productos financieros y sus tasas ofrecidos por esa entidad. Si pregunta por un producto, indica la entidad que ofrece la mejor tasa y cualquier observación relevante.
    """
    respuesta = ollama_llm.invoke(prompt)
    save_conversation(pregunta, respuesta)
    return respuesta

# Función principal del chatbot (Flan-T5)
def chatbot_response(user_input: str, qa_chain) -> str:
    if qa_chain is None: 
        return "Lo siento, el modelo de IA para riesgo de crédito no está disponible en este momento."

    input_lc = user_input.lower()
    if input_lc.startswith(("hola", "cómo estás", "como estas")):
        response = "¡Hola! Soy tu asistente especializado en riesgo de crédito en Colombia. ¿En qué te puedo ayudar?"
    elif "chiste" in input_lc:
        response = random.choice(financial_jokes)
    else:
        try:
            result = qa_chain({"query": user_input})
            response = result.get("result", "").strip()
            if not response:
                response = "No encontré información suficiente. ¿Podrías reformular tu pregunta?"
        except Exception as e:
            response = f"Hubo un error al procesar tu pregunta: {str(e)}"
    save_conversation(user_input, response)
    return response

# --- 4. Rutas de la Aplicación Flask ---

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        
        bot_response = chatbot_response(user_input, qa_chain)
        
        df_promedio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tasas_int.csv')
        bot_response_2 = "Error: Archivo 'tasas_int.csv' no encontrado."
        if os.path.exists(df_promedio_path):
            try:
                df_promedio = pd.read_csv(df_promedio_path)
                bot_response_2 = data_chat_bot_2(user_input, df_promedio)
            except Exception as e:
                bot_response_2 = f"Error al leer o procesar 'tasas_int.csv': {e}"
        else:
            flash("El archivo 'tasas_int.csv' no se encuentra en la carpeta de subidas. Por favor, súbelo.", "warning")

        # Añadimos 'selected_response_type' que será None por defecto
        chat_history.append({
            "user": user_input,
            "bot": bot_response,
            "bot_2": bot_response_2,
            "selected_response_type": None # Nuevo campo para guardar la selección
        })
    
    return render_template("index.html", history=chat_history)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No se encontró la parte del archivo en la solicitud.', 'error')
            return redirect(request.url)

        file = request.files['csv_file']

        if file.filename == '':
            flash('No se seleccionó ningún archivo.', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(file_path)
                flash(f'Archivo "{filename}" subido exitosamente!', 'success')
            except Exception as e:
                flash(f'Error al guardar el archivo: {e}', 'error')
            
            return redirect(url_for('index'))
        else:
            flash('Tipo de archivo no permitido. Solo se aceptan archivos .csv', 'error')
            return redirect(request.url)
    
    return redirect(url_for('index'))

@app.route('/elegir', methods=['POST'])
def elegir():
    # Eliminamos 'respuesta_elegida' ya que el HTML ya lo pasa por el name 'respuesta'
    # No es necesario extraerlo aquí si solo actualizaremos el estado y no lo mostraremos de nuevo
    
    chat_index_str = request.form.get('chat_index')
    response_type = request.form.get('response_type') # 'bot' o 'bot_2'

    if chat_index_str is not None and response_type is not None:
        try:
            chat_index = int(chat_index_str)
            if 0 <= chat_index < len(chat_history):
                # Actualizamos el campo 'selected_response_type' de la entrada de chat específica
                chat_history[chat_index]['selected_response_type'] = response_type
                # Opcional: Puedes mostrar un mensaje flash confirmando la selección
                flash(f'Has seleccionado una opción para la entrada de chat {chat_index + 1}.', 'info')
            else:
                flash('Índice de chat inválido.', 'warning')
        except ValueError:
            flash('Índice de chat no numérico.', 'error')
    else:
        flash('Faltan datos para seleccionar la respuesta.', 'warning')

    return redirect(url_for('index')) # Redirigimos a la página principal para que se refresque el chat

# --- 5. Ejecutar la Aplicación ---
if __name__ == '__main__':
    app.run(debug=True)