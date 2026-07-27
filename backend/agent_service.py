import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool, create_retriever_tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documentos"
INDEX_DIR = DATA_DIR / "indice_vectorial_vitalis"

load_dotenv(BASE_DIR / ".env")


SYSTEM_PROMPT = """
Eres el asistente virtual de Clínica Vitalis Salud.

Atiende exclusivamente consultas administrativas relacionadas con la clínica.

Reglas obligatorias:
1. Responde siempre en español, de forma clara, breve y profesional.
2. Para precios, utiliza la herramienta consultar_tarifa.
3. Para horarios, sedes y especialidades, utiliza consultar_turnos.
4. Para políticas, convenios, requisitos, cancelaciones, preparación para
   estudios e instrucciones, utiliza buscar_en_documentos_clinica.
5. No inventes precios, horarios, políticas, convenios ni datos médicos.
6. No diagnostiques, prescribas medicamentos ni sustituyas a un profesional.
7. Cuando no exista información suficiente, indícalo honestamente y sugiere
   comunicarse con Clínica Vitalis Salud al 722 555 0101.
"""


def normalizar_texto(texto: Any) -> str:
    """Normaliza texto para realizar búsquedas más tolerantes."""
    valor = str(texto).lower().strip()
    valor = unicodedata.normalize("NFD", valor)
    valor = "".join(
        caracter
        for caracter in valor
        if unicodedata.category(caracter) != "Mn"
    )
    valor = re.sub(r"[^a-z0-9\s]", " ", valor)
    return re.sub(r"\s+", " ", valor).strip()


def extraer_texto_respuesta(content: Any) -> str:
    """
    Convierte a texto tanto las respuestas tipo string como las respuestas
    estructuradas que devuelve Gemini.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        partes: list[str] = []

        for bloque in content:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict):
                texto = bloque.get("text")
                if texto:
                    partes.append(str(texto))

        return "\n".join(partes).strip()

    return str(content).strip()


class VitalisAgentService:
    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "No se encontró GOOGLE_API_KEY en las variables de entorno."
            )

        self.df_tarifas = self._cargar_csv("tarifas_consultas.csv")
        self.df_turnos = self._cargar_csv("especialidades_turnos.csv")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"No se encontró el índice FAISS en: {INDEX_DIR}"
            )

        self.vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=1024,
        )

        self.tools = self._crear_herramientas()

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
        )

    def _cargar_csv(self, nombre: str) -> pd.DataFrame:
        ruta = DOCUMENTS_DIR / nombre

        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

        return pd.read_csv(ruta)

    def consultar_tarifa(self, concepto: str) -> str:
        consulta = normalizar_texto(concepto)
        palabras = [
            palabra
            for palabra in consulta.split()
            if len(palabra) > 2
        ]

        if not palabras:
            return "No se recibió un concepto válido para consultar."

        tabla = self.df_tarifas.copy()
        tabla["_normalizado"] = tabla["concepto"].map(normalizar_texto)

        # Puntúa cada registro según la cantidad de palabras coincidentes.
        tabla["_puntaje"] = tabla["_normalizado"].apply(
            lambda texto: sum(palabra in texto for palabra in palabras)
        )

        maximo = int(tabla["_puntaje"].max())

        if maximo == 0:
            return (
                f"No encontré ninguna tarifa que coincida con '{concepto}'."
            )

        resultado = tabla[tabla["_puntaje"] == maximo]

        filas = [
            (
                f"- {fila.concepto}: "
                f"${fila.precio_mxn} MXN ({fila.categoria})"
            )
            for fila in resultado.itertuples()
        ]

        return "\n".join(filas)

    def consultar_turnos(self, especialidad: str) -> str:
        consulta = normalizar_texto(especialidad)
        tabla = self.df_turnos.copy()

        tabla["_normalizado"] = tabla["especialidad"].map(
            normalizar_texto
        )

        resultado = tabla[
            tabla["_normalizado"].str.contains(
                re.escape(consulta),
                regex=True,
                na=False,
            )
        ]

        if resultado.empty:
            palabras = [
                palabra
                for palabra in consulta.split()
                if len(palabra) > 2
            ]

            tabla["_puntaje"] = tabla["_normalizado"].apply(
                lambda texto: sum(
                    palabra in texto for palabra in palabras
                )
            )

            maximo = int(tabla["_puntaje"].max())

            if maximo > 0:
                resultado = tabla[tabla["_puntaje"] == maximo]

        if resultado.empty:
            return (
                f"No encontré turnos para la especialidad "
                f"'{especialidad}'."
            )

        filas = [
            (
                f"- {fila.especialidad} en {fila.sede}: "
                f"{fila.dias_atencion}, horario {fila.horario}"
            )
            for fila in resultado.itertuples()
        ]

        return "\n".join(filas)

    def _crear_herramientas(self) -> list:
        retriever_tool = create_retriever_tool(
            self.retriever,
            name="buscar_en_documentos_clinica",
            description=(
                "Busca información en los documentos oficiales de la clínica. "
                "Úsala para políticas, requisitos, cancelaciones, convenios, "
                "FAQ e instrucciones antes o después de la consulta."
            ),
        )

        tarifas_tool = Tool(
            name="consultar_tarifa",
            func=self.consultar_tarifa,
            description=(
                "Obtiene el precio exacto en MXN de consultas o estudios. "
                "Recibe el nombre o una descripción breve del concepto."
            ),
        )

        turnos_tool = Tool(
            name="consultar_turnos",
            func=self.consultar_turnos,
            description=(
                "Consulta sedes, días y horarios de una especialidad médica. "
                "Recibe el nombre de la especialidad."
            ),
        )

        return [retriever_tool, tarifas_tool, turnos_tool]

    def responder(self, mensaje: str) -> str:
        pregunta = mensaje.strip()

        if not pregunta:
            raise ValueError("El mensaje no puede estar vacío.")

        resultado = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": pregunta,
                    }
                ]
            }
        )

        mensajes = resultado.get("messages", [])

        if not mensajes:
            raise RuntimeError("El agente no devolvió mensajes.")

        respuesta = extraer_texto_respuesta(mensajes[-1].content)

        if not respuesta:
            return (
                "No fue posible generar una respuesta. "
                "Comunícate con la clínica al 722 555 0101."
            )

        return respuesta
