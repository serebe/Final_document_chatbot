import streamlit as st
from langchain.llms import OpenAI
from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv

def main():

    load_dotenv()


    st.set_page_config(
        page_title="Comparador de tasas de interés: Colombia",
    )

    st.header("¿Qué quieres preguntar?")

    user_csv = st.file_uploader(
        "Sube un archivo CSV con las tasas de interés de los bancos en Colombia",
        type=["csv"],
    )

    if user_csv is not None:
        user_question = st.text_input(
            "Pregunta sobre las tasas de interés",
            placeholder="¿Cuál es la tasa de interés más baja para un producto de libre inversión?",
        )

        llm = OpenAI(temperature=0)
        agent = create_csv_agent(
            llm, user_csv, verbose=True, allow_dangerous_code=True, 
        )

        if user_question is not None and user_question.strip() != "":
            st.write(f'Tu pregunta es: "{user_question}"')

            response = agent.run(
                user_question,
            )

            st.write(response)


if __name__ == "__main__":
    main()