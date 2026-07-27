from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.agent_service import VitalisAgentService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vitalis-api")

service: VitalisAgentService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service

    logger.info("Inicializando agente de Clínica Vitalis...")
    service = VitalisAgentService()
    logger.info("Agente inicializado correctamente.")

    yield

    service = None


app = FastAPI(
    title="Clínica Vitalis AI API",
    version="1.0.0",
    description="API del asistente virtual de Clínica Vitalis Salud.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    mensaje: str = Field(
        min_length=2,
        max_length=1500,
        examples=["¿Cuánto cuesta una consulta general?"],
    )


class ChatResponse(BaseModel):
    respuesta: str


@app.get("/")
def root():
    return {
        "servicio": "Clínica Vitalis AI API",
        "estado": "activo",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_ready": service is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="El agente todavía no está disponible.",
        )

    try:
        respuesta = service.responder(request.mensaje)
        return ChatResponse(respuesta=respuesta)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception("Error procesando la consulta")

        raise HTTPException(
            status_code=500,
            detail="No fue posible procesar la consulta.",
        ) from error
