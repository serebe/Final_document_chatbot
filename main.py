import streamlit as st
import pandas as pd
import unicodedata
from langchain_community.llms import Ollama

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

# Normalizar texto
def normalizar(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return texto.strip()

# Buscar nombre real de columna
def encontrar_columna(df, nombre_objetivo):
    nombre_objetivo = normalizar(nombre_objetivo)
    for col in df.columns:
        if normalizar(col) == nombre_objetivo:
            return col
    raise KeyError(f"No se encuentra la columna '{nombre_objetivo}' en el archivo CSV.")

# Buscar coincidencias parciales robustas
def filtrar_productos(df, pregunta):
    pregunta_norm = normalizar(pregunta)
    for k, v in SINONIMOS.items():
        if k in pregunta_norm:
            pregunta_norm = pregunta_norm.replace(k, v)

    col_producto = encontrar_columna(df, 'producto de credito')
    productos = df[col_producto].dropna().unique()
    productos_norm = [(p, normalizar(p)) for p in productos]

    coincidencias = [p for p, pn in productos_norm if all(palabra in pn for palabra in pregunta_norm.split() if len(palabra) > 3)]

    if not coincidencias:
        col_tipo = encontrar_columna(df, 'tipo_de_credito')
        tipos = df[col_tipo].dropna().unique()
        tipos_norm = [(t, normalizar(t)) for t in tipos]
        tipo_match = [t for t, tn in tipos_norm if any(palabra in tn for palabra in pregunta_norm.split())]
        if tipo_match:
            return df[df[col_tipo].isin(tipo_match)], tipo_match[0]
        return pd.DataFrame(), None

    return df[df[col_producto].isin(coincidencias)], coincidencias[0]

def main():
    st.set_page_config(page_title="Chatbot financiero: tasas de interés en Colombia")
    st.title("📊 Chatbot financiero de tasas en Colombia")

    archivo = st.file_uploader("Sube el archivo CSV con las tasas de interés", type=["csv"])

    if archivo:
        df = pd.read_csv(archivo)
        st.write("Vista previa de los datos:")
        st.dataframe(df.head())

        pregunta = st.text_input("¿Qué quieres saber sobre las tasas?")

        if pregunta:
            try:
                df_filtrado, producto_detectado = filtrar_productos(df, pregunta)

                if df_filtrado.empty:
                    st.warning("No se encontraron datos relevantes para tu consulta.")
                    return

                col_fecha = encontrar_columna(df, 'by_month_fecha_corte')
                muestra = df_filtrado.sort_values(col_fecha, ascending=False).head(30).to_markdown(index=False)

                prompt = f"""
Eres un asistente financiero experto en tasas de interés en Colombia. A continuación tienes información sobre productos financieros.

Producto consultado: {producto_detectado if producto_detectado else 'Detectado por categoría'}

Consulta del usuario: "{pregunta}"

Datos relevantes:
{muestra}

Por favor, responde en español, de forma clara y explicativa. Indica cuál es la tasa más baja, cuál entidad la ofrece, y si hay observaciones relevantes.
"""

                llm = Ollama(model="mistral", temperature=0.7)
                respuesta = llm.invoke(prompt)
                st.success("Respuesta del modelo:")
                st.write(respuesta)
            except Exception as e:
                st.error(f"Ocurrió un error al generar la respuesta: {e}")

if __name__ == "__main__":
    main()
