import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
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

Tu función es responder exclusivamente consultas administrativas relacionadas
con la clínica usando la herramienta de documentos disponible.

Reglas obligatorias:

1. Responde siempre en español, de forma clara, breve y profesional.
2. Usa buscar_en_documentos_clinica para responder preguntas sobre:
   - políticas;
   - requisitos;
   - cancelaciones;
   - reagendamientos;
   - convenios;
   - aseguradoras;
   - preparación para estudios;
   - instrucciones antes o después de una consulta.
3. No inventes información que no aparezca en los documentos.
4. No diagnostiques, no prescribas medicamentos y no sustituyas a un médico.
5. Si no encuentras información suficiente, indícalo honestamente y sugiere
   comunicarse con Clínica Vitalis Salud al 722 555 0101.
"""


def normalizar_texto(texto: Any) -> str:
    """Convierte un texto a una forma uniforme para realizar búsquedas."""
    valor = str(texto).lower().strip()
    valor = unicodedata.normalize("NFD", valor)
    valor = "".join(
        caracter
        for caracter in valor
        if unicodedata.category(caracter) != "Mn"
    )
    valor = re.sub(r"[^a-z0-9\s]", " ", valor)
    return re.sub(r"\s+", " ", valor).strip()


def extraer_texto_respuesta(mensaje: Any) -> str:
    """
    Extrae texto de respuestas simples o estructuradas de LangChain/Gemini.
    """
    if mensaje is None:
        return ""

    texto_directo = getattr(mensaje, "text", None)

    if isinstance(texto_directo, str) and texto_directo.strip():
        return texto_directo.strip()

    try:
        bloques = mensaje.content_blocks
    except Exception:
        bloques = None

    if bloques:
        textos: list[str] = []

        for bloque in bloques:
            if isinstance(bloque, dict):
                texto = bloque.get("text")
            else:
                texto = getattr(bloque, "text", None)

            if texto:
                textos.append(str(texto))

        if textos:
            return "\n".join(textos).strip()

    contenido = getattr(mensaje, "content", mensaje)

    if isinstance(contenido, str):
        return contenido.strip()

    if isinstance(contenido, list):
        textos: list[str] = []

        for bloque in contenido:
            if isinstance(bloque, str):
                textos.append(bloque)

            elif isinstance(bloque, dict):
                texto = bloque.get("text")

                if texto:
                    textos.append(str(texto))

            else:
                texto = getattr(bloque, "text", None)

                if texto:
                    textos.append(str(texto))

        return "\n".join(textos).strip()

    if isinstance(contenido, dict):
        texto = contenido.get("text")

        if texto:
            return str(texto).strip()

    return ""


class VitalisAgentService:
    """
    Servicio principal de Clínica Vitalis.

    Enrutamiento:
    - tarifas → Pandas/CSV;
    - turnos → Pandas/CSV;
    - políticas y documentos → Gemini + FAISS.
    """

    TERMINOS_TARIFA = {
        "precio",
        "precios",
        "costo",
        "costos",
        "cuesta",
        "cuestan",
        "costar",
        "tarifa",
        "tarifas",
        "valor",
    }

    TERMINOS_TURNO = {
        "horario",
        "horarios",
        "turno",
        "turnos",
        "atiende",
        "atienden",
        "atencion",
        "sede",
        "sedes",
        "dias",
    }
    
    TERMINOS_SERVICIOS = {
    "servicio",
    "servicios",
    "especialidad",
    "especialidades",
    "ofrecen",
    "manejan",
    "disponibles",
}

    PALABRAS_IGNORAR = {
        "hola",
        "cuanto",
        "cuantos",
        "cuesta",
        "cuestan",
        "cual",
        "cuales",
        "precio",
        "precios",
        "costo",
        "costos",
        "tarifa",
        "tarifas",
        "horario",
        "horarios",
        "turno",
        "turnos",
        "valor",
        "quisiera",
        "quiero",
        "saber",
        "informacion",
        "informes",
        "por",
        "favor",
        "sobre",
        "de",
        "del",
        "la",
        "las",
        "el",
        "los",
        "una",
        "un",
        "es",
        "son",
        "sus",
        "mi",
    }

    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "No se encontró GOOGLE_API_KEY en las variables de entorno."
            )

        self.df_tarifas = self._cargar_csv("tarifas_consultas.csv")
        self.df_turnos = self._cargar_csv("especialidades_turnos.csv")

        self.df_tarifas["_normalizado"] = self.df_tarifas["concepto"].map(
            normalizar_texto
        )

        self.df_turnos["_normalizado"] = self.df_turnos[
            "especialidad"
        ].map(normalizar_texto)

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

        self.retriever_tool = create_retriever_tool(
            self.retriever,
            name="buscar_en_documentos_clinica",
            description=(
                "Busca información en documentos oficiales de la clínica. "
                "Úsala para políticas, requisitos, cancelaciones, convenios, "
                "aseguradoras, preguntas frecuentes e instrucciones."
            ),
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=1024,
        )

        # El agente usa únicamente la herramienta documental.
        # Tarifas y horarios se resuelven directamente con Pandas.
        self.agent = create_agent(
            model=self.llm,
            tools=[self.retriever_tool],
            system_prompt=SYSTEM_PROMPT,
        )

    def _cargar_csv(self, nombre: str) -> pd.DataFrame:
        ruta = DOCUMENTS_DIR / nombre

        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

        return pd.read_csv(ruta)

    def _obtener_palabras_utiles(self, texto: str) -> list[str]:
        normalizado = normalizar_texto(texto)

        return [
            palabra
            for palabra in normalizado.split()
            if len(palabra) > 2
            and palabra not in self.PALABRAS_IGNORAR
        ]

    def _contiene_termino(
        self,
        texto: str,
        terminos: set[str],
    ) -> bool:
        palabras = set(normalizar_texto(texto).split())
        return bool(palabras.intersection(terminos))
    
    def listar_servicios(self) -> str:
        """Lista las especialidades disponibles sin consultar Gemini."""
        servicios = sorted(
            self.df_turnos["especialidad"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if not servicios:
            return (
                "No encontré especialidades registradas. "
                "Comunícate con la clínica al 722 555 0101."
            )

        lista = "\n".join(
            f"- **{servicio}**"
            for servicio in servicios
        )

        return (
            "Estas son las especialidades disponibles:\n\n"
            f"{lista}\n\n"
            "Indícame cuál deseas consultar para revisar su tarifa "
            "u horario de atención."
        )
    
    def _es_respuesta_contextual(
        self,
        pregunta: str,
        ultima_intencion: Optional[str],
    ) -> bool:
        """
        Comprueba si el mensaje corresponde realmente
        a la intención anterior.
        """
        if ultima_intencion == "tarifa":
            return self._parece_concepto_tarifa(pregunta)

        if ultima_intencion == "turno":
            return self._parece_especialidad(pregunta)

        return False

    def _aplicar_alias_tarifa(self, consulta: str) -> str:
        alias = {
            "consulta medica general": "consulta medicina general",
            "consulta medica": "consulta medicina general",
            "consulta general": "consulta medicina general",
            "medicina general": "consulta medicina general",
            "medico general": "consulta medicina general",
            "doctor general": "consulta medicina general",
            "pediatra": "consulta pediatria",
            "cardiologo": "consulta cardiologia",
            "nutriologo": "consulta nutricion",
            "psicologo": "consulta psicologia",
            "ginecologo": "consulta ginecologia",
            "laboratorio basico": "estudio de laboratorio basico",
            "ultrasonido pelvico": "ultrasonido pelvico",
            "ultrasonido abdominal": "ultrasonido abdominal",
            "radiografia": "radiografia simple",
        }

        consulta_normalizada = normalizar_texto(consulta)

        for expresion, concepto in alias.items():
            if expresion in consulta_normalizada:
                return concepto

        return consulta_normalizada

    def _aplicar_alias_turno(self, consulta: str) -> str:
        alias = {
            "medico general": "medicina general",
            "doctor general": "medicina general",
            "consulta general": "medicina general",
            "pediatra": "pediatria",
            "cardiologo": "cardiologia",
            "nutriologo": "nutricion",
            "psicologo": "psicologia",
            "ginecologo": "ginecologia",
        }

        consulta_normalizada = normalizar_texto(consulta)

        for expresion, especialidad in alias.items():
            if expresion in consulta_normalizada:
                return especialidad

        return consulta_normalizada
         
    def listar_tarifas(self) -> str:
        """Lista todas las tarifas registradas sin consultar Gemini."""
        if self.df_tarifas.empty:
            return (
                "No encontré tarifas registradas. "
                "Comunícate con la clínica al 722 555 0101."
            )

        filas = [
            (
                f"- **{fila.concepto}**  \n"
                f"Precio: **${fila.precio_mxn:,.0f} MXN**  \n"
                f"Categoría: {fila.categoria}"
            )
            for fila in self.df_tarifas.itertuples()
        ]

        return (
            "Estas son las tarifas disponibles:\n\n"
            + "\n\n".join(filas)
        )
        
    def consultar_tarifa(self, concepto: str) -> str:
        """Consulta precios exactos en el CSV sin llamar a Gemini."""
        consulta = self._aplicar_alias_tarifa(concepto)
        palabras = self._obtener_palabras_utiles(consulta)
                
        if not palabras:
            return self.listar_tarifas()

        tabla = self.df_tarifas.copy()

        tabla["_puntaje"] = tabla["_normalizado"].apply(
            lambda texto: sum(
                palabra in texto
                for palabra in palabras
            )
        )

        maximo = int(tabla["_puntaje"].max())

        if maximo == 0:
            return (
                f"No encontré una tarifa que coincida con **{concepto}**. "
                "Comunícate con la clínica al 722 555 0101."
            )

        resultado = tabla[tabla["_puntaje"] == maximo]

        # Evita listar toda la tabla cuando la consulta es demasiado genérica.
        if len(resultado) > 4:
            return (
                "Encontré varias consultas. Indícame la especialidad o "
                "el estudio exacto que deseas cotizar."
            )

        filas = [
            (
                f"**{fila.concepto}**  \n"
                f"Precio: **${fila.precio_mxn:,.0f} MXN**  \n"
                f"Categoría: {fila.categoria}"
            )
            for fila in resultado.itertuples()
        ]

        return "\n\n".join(filas)

    def consultar_turnos(self, especialidad: str) -> str:
        """Consulta sedes, días y horarios sin llamar a Gemini."""
        consulta = self._aplicar_alias_turno(especialidad)
        palabras = self._obtener_palabras_utiles(consulta)

        if not palabras:
            filas = [
            (
                f"- **{fila.especialidad}** "
                f"({fila.sede})\n"
                f"{fila.dias_atencion} | {fila.horario}"
            )
            for fila in self.df_turnos.itertuples()
        ]

            return (
                "Estos son nuestros horarios disponibles:\n\n"
                + "\n\n".join(filas)
            )

        tabla = self.df_turnos.copy()

        tabla["_puntaje"] = tabla["_normalizado"].apply(
            lambda texto: sum(
                palabra in texto
                for palabra in palabras
            )
        )

        maximo = int(tabla["_puntaje"].max())

        if maximo == 0:
            return (
                f"No encontré horarios para **{especialidad}**. "
                "Comunícate con la clínica al 722 555 0101."
            )

        resultado = tabla[tabla["_puntaje"] == maximo]

        filas = [
            (
                f"- **{fila.especialidad}**, sede **{fila.sede}**: "
                f"{fila.dias_atencion}, horario {fila.horario}."
            )
            for fila in resultado.itertuples()
        ]

        return "\n".join(filas)

    def _parece_concepto_tarifa(self, pregunta: str) -> bool:
        consulta = self._aplicar_alias_tarifa(pregunta)
        palabras = self._obtener_palabras_utiles(consulta)

        if not palabras:
            return False

        return any(
            any(palabra in concepto for palabra in palabras)
            for concepto in self.df_tarifas["_normalizado"]
        )

    def _parece_especialidad(self, pregunta: str) -> bool:
        consulta = self._aplicar_alias_turno(pregunta)
        palabras = self._obtener_palabras_utiles(consulta)

        if not palabras:
            return False

        return any(
            any(palabra in especialidad for palabra in palabras)
            for especialidad in self.df_turnos["_normalizado"]
        )

    def _responder_documentos(self, pregunta: str) -> str:
        """Ejecuta Gemini únicamente para documentos y políticas."""
        try:
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

            respuesta = extraer_texto_respuesta(mensajes[-1])

            if not respuesta:
                return (
                    "No encontré una respuesta en los documentos disponibles. "
                    "Comunícate con la clínica al 722 555 0101."
                )

            return respuesta

        except Exception as error:
            detalle = str(error)

            if "RESOURCE_EXHAUSTED" in detalle or "429" in detalle:
                return (
                    "El servicio de consulta documental alcanzó temporalmente "
                    "su límite de uso. Las tarifas y horarios siguen "
                    "disponibles. Intenta nuevamente más tarde."
                )

            return (
                "No fue posible consultar los documentos en este momento. "
                "Comunícate con la clínica al 722 555 0101."
            )

    def responder(
        self,
        mensaje: str,
        ultima_intencion: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Responde y devuelve:
        - texto de respuesta;
        - intención que debe conservar Streamlit.
        """
        pregunta = mensaje.strip()

        if not pregunta:
            raise ValueError("El mensaje no puede estar vacío.")

        pregunta_normalizada = normalizar_texto(pregunta)

        # Saludos simples: no consumen Gemini.
        if pregunta_normalizada in {
            "hola",
            "buen dia",
            "buenos dias",
            "buenas tardes",
            "buenas noches",
        }:
            return (
                "Hola. Puedo ayudarte con tarifas, horarios, requisitos, "
                "políticas y convenios de Clínica Vitalis Salud.",
                None,
            )

        es_tarifa_explicita = self._contiene_termino(
            pregunta,
            self.TERMINOS_TARIFA,
        )

        es_turno_explicito = self._contiene_termino(
            pregunta,
            self.TERMINOS_TURNO,
        )
        
        es_consulta_servicios = (
            self._contiene_termino(
                pregunta,
                self.TERMINOS_SERVICIOS,
            )
            or pregunta_normalizada in {
                "cuales hay",
                "que hay",
                "que ofrecen",
                "que manejan",
            }
        )
        
        # Pregunta general sobre servicios o especialidades.
        if es_consulta_servicios:
            return self.listar_servicios(), None

        # El usuario cambia explícitamente a tarifas.
        if es_tarifa_explicita:
            respuesta = self.consultar_tarifa(pregunta)
            return respuesta, "tarifa"

        # El usuario cambia explícitamente a horarios.
        if es_turno_explicito:
            respuesta = self.consultar_turnos(pregunta)
            return respuesta, "turno"

        # Solo conserva el contexto cuando el mensaje coincide
        # con una entidad válida para la intención anterior.
        if self._es_respuesta_contextual(
            pregunta,
            ultima_intencion,
        ):
            if ultima_intencion == "tarifa":
                return self.consultar_tarifa(pregunta), "tarifa"

            if ultima_intencion == "turno":
                return self.consultar_turnos(pregunta), "turno"

        # Concepto reconocido, pero sin saber si desea precio u horario.
        parece_tarifa = self._parece_concepto_tarifa(pregunta)
        parece_turno = self._parece_especialidad(pregunta)

        if parece_tarifa and parece_turno:
            return (
                "Encontré esa especialidad. ¿Deseas consultar su "
                "**tarifa** o su **horario de atención**?",
                None,
            )

        # Todo lo demás se consulta en documentos.
        respuesta = self._responder_documentos(pregunta)
        return respuesta, None
