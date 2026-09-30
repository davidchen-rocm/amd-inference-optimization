"""Loopback-only HTTP contract; gateway protection and verified Firebase ownership."""

from __future__ import annotations

import asyncio
import hmac
import json
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from macfit_training.config import capabilities, validate_job_input

from .auth import AuthenticationError, AuthenticationUnavailable, FirebaseVerifier, Identity
from .outputs import artifact_handle
from .settings import Settings
from .store import JobStore, ServiceError, public_job
from .supervisor import Supervisor

PREFIX = "/api/training"


def error_response(status: int, code: str, message: str, retryable: bool = False):
    return JSONResponse(
        {"error": {"code": code, "message": message, "retryable": retryable}},
        status_code=status,
        headers={"Cache-Control": "no-store", **({"Retry-After": "5"} if retryable else {})},
    )


class GatewayBodyMiddleware:
    def __init__(self, app, *, secret: str, maximum: int):
        self.app, self.secret, self.maximum = app, secret.encode(), maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        pairs = scope.get("headers", [])
        gateways = [value for name, value in pairs if name.lower() == b"x-training-gateway"]
        if len(gateways) != 1 or not hmac.compare_digest(gateways[0], self.secret):
            return await error_response(
                403, "gateway_required", "This service is available through the MacFit website."
            )(scope, receive, send)
        lengths = [value for name, value in pairs if name.lower() == b"content-length"]
        if len(lengths) > 1 or (lengths and (not lengths[0].isdigit() or len(lengths[0]) > 12)):
            return await error_response(400, "invalid_request", "The request size is invalid.")(
                scope, receive, send
            )
        if lengths and int(lengths[0]) > self.maximum:
            return await error_response(
                413, "request_too_large", "Use a request smaller than 3 MiB."
            )(scope, receive, send)
        if scope["method"] in {"POST", "PUT", "PATCH", "DELETE"}:
            content = bytearray()
            try:
                async with asyncio.timeout(15):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        if message["type"] != "http.request":
                            continue
                        data = message.get("body", b"")
                        if len(content) + len(data) > self.maximum:
                            return await error_response(
                                413, "request_too_large", "Use a request smaller than 3 MiB."
                            )(scope, receive, send)
                        content.extend(data)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await error_response(
                    408, "request_timeout", "The upload took too long. Please try again."
                )(scope, receive, send)
            used = False

            async def replay():
                nonlocal used
                if not used:
                    used = True
                    return {"type": "http.request", "body": bytes(content), "more_body": False}
                return await receive()

            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)


def parse_uuid(value: str, name: str) -> str:
    try:
        if not isinstance(value, str):
            raise ValueError("not text")
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise ServiceError(422, "invalid_request", f"{name} must be a UUID.") from error


def create_app(
    settings: Settings | None = None,
    *,
    verifier=None,
    validator: Callable = validate_job_input,
    capability_provider: Callable = capabilities,
    readiness: Callable[[], bool] | None = None,
    supervisor_factory: Callable = Supervisor,
) -> FastAPI:
    settings = settings or Settings.from_env()
    verifier = verifier or FirebaseVerifier(settings.firebase_project_id)
    readiness = readiness or settings.gpu_ready
    store = JobStore(settings)
    supervisor = supervisor_factory(store, settings, readiness=readiness)

    @asynccontextmanager
    async def lifespan(_app):
        await run_in_threadpool(supervisor.start)
        try:
            yield
        finally:
            await run_in_threadpool(supervisor.stop)

    app = FastAPI(
        title="MacFit training", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.add_middleware(
        GatewayBodyMiddleware, secret=settings.gateway_secret, maximum=settings.max_body_bytes
    )
    app.state.store, app.state.supervisor = store, supervisor

    @app.exception_handler(ServiceError)
    async def service_error(_request, error):
        return error_response(error.status, error.code, error.message, error.retryable)

    @app.exception_handler(Exception)
    async def unexpected_error(_request, _error):
        # Do not leak paths, datasets, credentials, SQLite errors or subprocess details.
        return error_response(
            500, "internal_error", "The request could not be completed. Please try again.", True
        )

    def identity(request: Request) -> Identity:
        headers = [
            value for name, value in request.scope["headers"] if name.lower() == b"authorization"
        ]
        if len(headers) != 1:
            raise ServiceError(401, "sign_in_required", "Sign in to use your training jobs.")
        try:
            value = headers[0].decode("ascii")
            scheme, token = value.split(" ", 1)
            if scheme.lower() != "bearer" or not token or " " in token:
                raise AuthenticationError("Invalid sign-in token")
            resolved = verifier.verify(token)
            if not isinstance(resolved, Identity) or not resolved.uid:
                raise AuthenticationError("Invalid verified identity")
            return resolved
        except AuthenticationUnavailable as error:
            raise ServiceError(
                503,
                "authentication_unavailable",
                "Sign-in verification is temporarily unavailable. Please try again.",
                True,
            ) from error
        except (AuthenticationError, ValueError, UnicodeError) as error:
            raise ServiceError(
                401, "sign_in_required", "Your sign-in has expired. Please sign in again."
            ) from error

    owner_dependency = Depends(identity)

    def available() -> bool:
        return supervisor.healthy and readiness() and settings.accepts_new()

    @app.get(PREFIX + "/capabilities")
    def get_capabilities():
        payload = capability_provider()
        payload["available"] = available()
        payload["auth"] = {"required": True, "provider": "firebase"}
        payload["queue"] = {"active": store.queue_size(), "capacity": settings.max_queue_jobs}
        if not payload["available"]:
            payload["reason"] = (
                "New GPU jobs are paused. Your saved jobs and downloads remain available."
                if not settings.accepts_new()
                else "The GPU training service is not ready. Please try again later."
            )
        payload["limits"] = {
            **payload.get("limits", {}),
            "max_body_bytes": settings.max_body_bytes,
            "max_active_jobs_per_owner": 1,
            "max_queued_jobs": settings.max_queue_jobs,
        }
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @app.get("/livez")
    def live():
        if not supervisor.healthy:
            raise ServiceError(
                503, "worker_unavailable", "The training worker is unavailable.", True
            )
        return {"status": "ok"}

    @app.post(PREFIX + "/jobs")
    async def submit_job(request: Request, owner: Identity = owner_dependency):
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            raise ServiceError(415, "invalid_content_type", "Send the job as JSON.")
        try:
            body = json.loads(
                await request.body(),
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")),
            )
            if not isinstance(body, dict) or set(body) != {
                "request_id",
                "project_id",
                "kind",
                "input",
            }:
                raise ValueError("Provide request_id, project_id, kind and input only.")
            request_id = parse_uuid(body["request_id"], "request_id")
            project_id = parse_uuid(body["project_id"], "project_id")
            if body["kind"] not in ("generation", "training"):
                raise ValueError("Choose generation or training.")
            normalized = await run_in_threadpool(validator, body["kind"], body["input"])
        except (ValueError, TypeError, UnicodeError, RecursionError) as error:
            message = (
                str(error)
                if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError)
                else "The job input is not valid JSON."
            )
            raise ServiceError(422, "invalid_input", message[:400]) from error
        # An ambiguous prior response remains recoverable during a GPU outage.
        prior = await run_in_threadpool(store.list, owner.uid, request_id=request_id)
        if not prior and not settings.accepts_new():
            raise ServiceError(
                503,
                "gpu_window_closed",
                "New GPU jobs are paused. Your saved jobs and downloads remain available.",
            )
        if not prior and not available():
            raise ServiceError(
                503,
                "worker_unavailable",
                "The GPU training service is not ready. Please try again later.",
                True,
            )
        job, created = await run_in_threadpool(
            store.create, owner.uid, request_id, project_id, body["kind"], normalized
        )
        if created:
            supervisor.notify()
        return JSONResponse(
            public_job(job),
            status_code=202 if created else 200,
            headers={"Cache-Control": "no-store"},
        )

    @app.get(PREFIX + "/jobs")
    def list_jobs(request: Request, owner: Identity = owner_dependency):
        if set(request.query_params) - {"project_id", "request_id"} or any(
            len(request.query_params.getlist(key)) != 1 for key in request.query_params
        ):
            raise ServiceError(422, "invalid_request", "Use project_id or request_id to find jobs.")
        project_id = (
            parse_uuid(request.query_params["project_id"], "project_id")
            if "project_id" in request.query_params
            else None
        )
        request_id = (
            parse_uuid(request.query_params["request_id"], "request_id")
            if "request_id" in request.query_params
            else None
        )
        return JSONResponse(
            {
                "items": [
                    public_job(job)
                    for job in store.list(owner.uid, project_id=project_id, request_id=request_id)
                ]
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get(PREFIX + "/jobs/{job_id}")
    def get_job(job_id: str, owner: Identity = owner_dependency):
        return JSONResponse(
            public_job(store.get(job_id, owner.uid)), headers={"Cache-Control": "no-store"}
        )

    @app.post(PREFIX + "/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, owner: Identity = owner_dependency):
        job = store.cancel(job_id, owner.uid)
        supervisor.notify()
        return JSONResponse(
            public_job(job),
            status_code=202 if job["status"] == "cancelling" else 200,
            headers={"Cache-Control": "no-store"},
        )

    @app.get(PREFIX + "/jobs/{job_id}/artifacts/{artifact_id}")
    def download_artifact(job_id: str, artifact_id: str, owner: Identity = owner_dependency):
        job = store.get(job_id, owner.uid)
        artifact = next((item for item in job["artifacts"] if item["id"] == artifact_id), None)
        if job["status"] != "succeeded" or artifact is None:
            raise ServiceError(404, "not_found", "This download could not be found.")
        handle = artifact_handle(store.directory(job_id), artifact, settings)

        def chunks():
            try:
                while chunk := handle.read(1024 * 1024):
                    yield chunk
            finally:
                handle.close()

        return StreamingResponse(
            chunks(),
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(artifact["size_bytes"]),
                "Content-Disposition": f'attachment; filename="{artifact["name"]}"',
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "ETag": '"' + artifact["sha256"] + '"',
            },
            background=BackgroundTask(handle.close),
        )

    return app
