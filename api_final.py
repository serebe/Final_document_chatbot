from pathlib import Path
import pandas as pd
import unicodedata
from langchain_community.llms import Ollama
import socket

# Diccionario de sinónimos simples
SINONIMOS = {
    "hipotecario": "vivienda",
    "crédito hipotecario": "vivienda",
    "credito hipotecario": "vivienda",
    "tarjeta": "tarjeta de crédito",
    "tarjeta de credito": "tarjeta de crédito",
    "carro": "vehículo",
    "auto": "vehículo",
    "libre": "libre inversión",
    "microcreditos": "microcrédito",
    "microcredito": "microcrédito",
    "educativo": "créditos educativos diferentes a libranza"
}

# --- Utilidades ---
def normalizar(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return texto.strip()

def encontrar_columna(df, nombre_objetivo):
    nombre_objetivo = normalizar(nombre_objetivo)
    for col in df.columns:
        if normalizar(col) == nombre_objetivo:
            return col
    sugerencias = [col for col in df.columns if nombre_objetivo in normalizar(col)]
    if sugerencias:
        raise KeyError(f"No se encuentra la columna '{nombre_objetivo}'. ¿Quisiste decir: {sugerencias}?")
    raise KeyError(f"No se encuentra la columna '{nombre_objetivo}' en el archivo CSV.")

def filtrar_productos(df, pregunta):
    pregunta_norm = normalizar(pregunta)
    for k, v in SINONIMOS.items():
        if k in pregunta_norm:
            pregunta_norm = pregunta_norm.replace(k, v)

    col_producto = encontrar_columna(df, 'Producto de crédito')
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

def check_ollama_connection(host="localhost", port=11434):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False

def data_chat_bot_2(pregunta: str, df: pd.DataFrame):
    try:
        llm = Ollama(model="mistral", temperature=0.7)
    except Exception as e:
        return f"❌ No se pudo conectar con Ollama: {e}"

    df_filtrado, producto_detectado = filtrar_productos(df, pregunta)

    if df_filtrado.empty:
        return "⚠️ No encontré datos relevantes para tu consulta. ¿Podrías ser más específico?"

    try:
        col_fecha = encontrar_columna(df, 'by_month_fecha_corte')
    except Exception as e:
        return str(e)

    if col_fecha not in df_filtrado.columns:
        return f"⚠️ Error: La columna de fecha '{col_fecha}' no se encontró en los datos filtrados."

    muestra = df_filtrado.sort_values(col_fecha, ascending=False).head(30).to_markdown(index=False)

    prompt = f"""
Eres un asistente financiero experto en tasas de interés en Colombia. A continuación tienes información sobre productos financieros.
Consulta del usuario: "{pregunta}"
Datos relevantes:
{muestra}
Por favor, responde en español, de forma clara y explicativa. Si el usuario pregunta por una entidad, indica cuáles son los productos financieros y sus tasas ofrecidos por esa entidad. Si pregunta por un producto, asegúrate de que el resultado corresponda al producto indicado, y no a otro producto distinto aunque tenga una tasa más baja. Si la consulta menciona tipo de persona (natural o jurídica), tenlo en cuenta al seleccionar los resultados.
"""

    try:
        respuesta = llm.invoke(prompt)
    except Exception as e:
        return f"❌ Error al generar respuesta del modelo: {e}"

    return respuesta

# --- EJECUCIÓN CONSOLE ---
if __name__ == "__main__":
    print("📊 Asistente financiero de tasas de interés en Colombia 🇨🇴")

    if not check_ollama_connection():
        print("❌ Ollama no está corriendo en http://localhost:11434")
        print("➡️ Inicia el servidor con: `ollama serve` o `ollama run mistral`")
        exit()

    csv_path = Path("upload/tasas_int.csv")
    if not csv_path.exists():
        print("⚠️ Archivo CSV no encontrado en 'upload/tasas_int.csv'")
        exit()

    try:
        data = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        print(f"❌ Error al leer el CSV: {e}")
        exit()

    while True:
        pregunta = input("\n🧑 Usuario: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("👋 Hasta luego.")
            break
        respuesta = data_chat_bot_2(pregunta, data)
        print(f"\n🤖 Chatbot:\n{respuesta}")
