from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_service.errors import RagEmbeddingError, RagIndexNotReadyError
from rag_service.models import RetrievalRequest, RetrievalResponse
from rag_service.service import RagService


def create_router(service: RagService, *, api_token: str | None = None) -> APIRouter:
    router = APIRouter(prefix="/v1/rag", tags=["Standalone RAG"])

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if api_token is None:
            return
        expected = f"Bearer {api_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    @router.get("/health")
    async def health() -> dict:
        payload = {"status": "ok", "backend": service.backend.name}
        status = getattr(service.backend, "status", None)
        if status is not None:
            try:
                payload["capabilities"] = status()
            except Exception as exc:  # health must never fail on a probe
                payload["capabilities"] = {"error": type(exc).__name__}
        return payload

    @router.get("/categories")
    async def categories(knowledge_base: str = "cybersec") -> dict:
        try:
            values = service.config.content_categories(knowledge_base)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"knowledge_base": knowledge_base, "categories": values}

    @router.post("/search", response_model=RetrievalResponse, dependencies=[Depends(authorize)])
    async def search(request: RetrievalRequest) -> RetrievalResponse:
        try:
            return service.search(request)
        except RagIndexNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RagEmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


def create_app(service: RagService, *, api_token: str | None = None) -> FastAPI:
    app = FastAPI(title="Standalone RAG Retrieval API", version="1.0")
    app.include_router(create_router(service, api_token=api_token))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        # FastAPI's default 422 echoes the offending value verbatim, so a long
        # query paste lands in logs (and any client context) in full. Keep the
        # structured detail but drop the echoed input from `loc`-level messages.
        from rag_service.models import describe_validation_error

        return JSONResponse(
            status_code=422,
            content={"detail": f"invalid request: {describe_validation_error(exc)}"},
        )

    return app
