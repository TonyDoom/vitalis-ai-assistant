import os
import sys
from pathlib import Path

import streamlit as st


# Agregar la raíz del repositorio al PATH de Python.
# Esto permite importar backend.agent_service cuando Streamlit
# ejecuta el archivo ubicado dentro de frontend/.
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


st.set_page_config(
    page_title="Clínica Vitalis Salud",
    page_icon="🏥",
    layout="centered",
)


def configurar_api_key() -> None:
    """
    Obtiene GOOGLE_API_KEY desde Streamlit Secrets.
    Como respaldo, permite utilizar una variable de entorno local.
    """
    if os.getenv("GOOGLE_API_KEY"):
        return

    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = None

    if not api_key:
        st.error(
            "No se encontró GOOGLE_API_KEY. "
            "Configúrala en Streamlit Secrets."
        )
        st.stop()

    os.environ["GOOGLE_API_KEY"] = str(api_key)


configurar_api_key()

# Se importa después de configurar la API Key.
from backend.agent_service import VitalisAgentService  # noqa: E402


@st.cache_resource(show_spinner="Inicializando asistente de Vitalis...")
def cargar_agente() -> VitalisAgentService:
    """
    Carga una sola instancia del agente, FAISS, Gemini y los CSV.
    Streamlit reutiliza esta instancia entre consultas.
    """
    return VitalisAgentService()


st.title("🏥 Clínica Vitalis Salud")
st.caption(
    "Asistente para consultar tarifas, horarios, requisitos, "
    "políticas e información administrativa."
)

with st.sidebar:
    st.subheader("Acerca del asistente")

    st.write(
        "La información se obtiene de documentos internos, "
        "tarifas y horarios de Clínica Vitalis Salud."
    )

    st.warning(
        "Este asistente no realiza diagnósticos, no prescribe "
        "medicamentos y no sustituye una consulta médica."
    )

    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


try:
    agente = cargar_agente()
except Exception as error:
    st.error("No fue posible inicializar el asistente.")
    st.exception(error)
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hola. Soy el asistente virtual de Clínica Vitalis Salud. "
                "Puedo ayudarte con tarifas, horarios, requisitos y políticas."
            ),
        }
    ]


for mensaje in st.session_state.messages:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])


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
                respuesta = agente.responder(pregunta)

            except Exception as error:
                respuesta = (
                    "No fue posible procesar la consulta. "
                    "Intenta nuevamente o comunícate con la clínica "
                    "al 722 555 0101."
                )

                # El detalle aparece en pantalla para facilitar
                # el diagnóstico durante la entrega académica.
                st.error(f"Detalle técnico: {error}")

        st.markdown(respuesta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta,
        }
    )