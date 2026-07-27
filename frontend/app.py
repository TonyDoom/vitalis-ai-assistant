import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
).rstrip("/")


st.set_page_config(
    page_title="Clínica Vitalis Salud",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Clínica Vitalis Salud")
st.caption(
    "Asistente para tarifas, horarios, requisitos, políticas e información "
    "administrativa."
)

with st.sidebar:
    st.subheader("Información")
    st.write(
        "Este asistente proporciona información administrativa basada "
        "en documentos internos."
    )
    st.warning(
        "No realiza diagnósticos ni reemplaza una consulta médica."
    )

    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hola. Soy el asistente virtual de Clínica Vitalis Salud. "
                "¿En qué puedo ayudarte?"
            ),
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


pregunta = st.chat_input(
    "Pregunta por una tarifa, especialidad, requisito o política..."
)


if pregunta:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": pregunta,
        }
    )

    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando información..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"mensaje": pregunta},
                    timeout=120,
                )

                response.raise_for_status()
                respuesta = response.json()["respuesta"]

            except requests.exceptions.ConnectionError:
                respuesta = (
                    "No fue posible conectar con el servidor de la clínica."
                )

            except requests.exceptions.Timeout:
                respuesta = (
                    "La consulta tardó demasiado tiempo. Intenta nuevamente."
                )

            except requests.exceptions.HTTPError:
                try:
                    detalle = response.json().get(
                        "detail",
                        "Error desconocido",
                    )
                except Exception:
                    detalle = "Error desconocido"

                respuesta = f"El servidor reportó un error: {detalle}"

            except Exception:
                respuesta = (
                    "Ocurrió un error inesperado al procesar la consulta."
                )

        st.markdown(respuesta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta,
        }
    )
