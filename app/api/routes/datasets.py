from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Annotated, Any, TypeVar
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.routing import APIRoute
from pydantic import BaseModel
from python_multipart.exceptions import MultipartParseError
from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)
from starlette.formparsers import MultiPartException

from app.services.dataset_service import (
    MAX_CSV_BYTES,
    DatasetIntegrityError,
    DatasetMetadata,
    DatasetService,
)


logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT")

# Allow a small amount of multipart overhead beyond the CSV limit.
MAX_UPLOAD_BODY_BYTES = MAX_CSV_BYTES + 64 * 1024


class DatasetErrorDetail(BaseModel):
    code: str
    message: str


class DatasetErrorResponse(BaseModel):
    detail: DatasetErrorDetail


def _http_error(
    status_code: int,
    code: str,
    message: str,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
        },
        headers={
            "Cache-Control": "no-store",
        },
    )


def _upload_too_large() -> HTTPException:
    return _http_error(
        413,
        "upload_too_large",
        "Upload one CSV of at most 25 MB.",
    )


class DatasetUploadRoute(APIRoute):
    """Limit request bodies before FastAPI parses file parameters."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            if request.method != "POST":
                return await original(request)

            media_type = request.headers.get(
                "content-type",
                "",
            ).split(";", 1)[0]

            if media_type.strip().lower() != "multipart/form-data":
                raise _http_error(
                    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    "multipart_required",
                    "Send the CSV as multipart/form-data "
                    "in the file field.",
                )

            declared = request.headers.get(
                "content-length"
            )

            if declared is not None:
                if (
                    not declared.isascii()
                    or not declared.isdecimal()
                ):
                    raise _http_error(
                        400,
                        "invalid_content_length",
                        "Invalid Content-Length header.",
                    )

                try:
                    declared_size = int(declared)

                except ValueError:
                    raise _upload_too_large() from None

                if declared_size > MAX_UPLOAD_BODY_BYTES:
                    raise _upload_too_large()

            received = 0
            exceeded = False

            async def limited_receive():
                nonlocal received, exceeded

                message = await request.receive()

                if message["type"] == "http.request":
                    received += len(
                        message.get("body", b"")
                    )

                    if received > MAX_UPLOAD_BODY_BYTES:
                        exceeded = True

                        # Multipart parsers close temporary files
                        # when this exception is raised.
                        raise MultiPartException(
                            "Upload body limit exceeded."
                        )

                return message

            bounded = Request(
                request.scope,
                receive=limited_receive,
            )

            try:
                # Cache the parsed form for FastAPI's UploadFile parameter.
                form = await bounded.form(
                    max_files=1,
                    max_fields=0,
                )

            except StarletteHTTPException as exc:
                if exceeded:
                    raise _upload_too_large() from None

                if (
                    exc.status_code == 400
                    and isinstance(exc.detail, str)
                ):
                    raise _http_error(
                        400,
                        "invalid_multipart",
                        "Provide one complete CSV upload "
                        "in the file field.",
                    ) from None

                raise

            except (
                MultiPartException,
                MultipartParseError,
                UnicodeError,
                LookupError,
            ):
                if exceeded:
                    raise _upload_too_large() from None

                raise _http_error(
                    400,
                    "invalid_multipart",
                    "Invalid multipart upload.",
                ) from None

            try:
                if set(form.keys()) != {"file"}:
                    raise _http_error(
                        400,
                        "invalid_upload",
                        "Provide exactly one file field.",
                    )

                return await original(bounded)

            finally:
                await form.close()

        return handler


router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
    route_class=DatasetUploadRoute,
    responses={
        400: {
            "model": DatasetErrorResponse,
            "description": "Invalid CSV or upload.",
        },
        404: {
            "model": DatasetErrorResponse,
            "description": "Dataset not found.",
        },
        409: {
            "model": DatasetErrorResponse,
            "description": "Upload identity conflict.",
        },
        413: {
            "model": DatasetErrorResponse,
            "description": "Upload exceeds the size limit.",
        },
        415: {
            "model": DatasetErrorResponse,
            "description": "Unsupported upload format.",
        },
        500: {
            "model": DatasetErrorResponse,
            "description": "Dataset integrity or server error.",
        },
        503: {
            "model": DatasetErrorResponse,
            "description": "Dataset storage is unavailable.",
        },
    },
)


async def get_dataset_service(
    request: Request,
    response: Response,
) -> DatasetService:
    """The application's lifespan supplies this shared instance."""
    response.headers["Cache-Control"] = "no-store"

    service = getattr(
        request.app.state,
        "dataset_service",
        None,
    )

    if service is None:
        raise _http_error(
            503,
            "dataset_service_unavailable",
            "Dataset service is not ready. Try again shortly.",
        )

    return service


ServiceDependency = Annotated[
    DatasetService,
    Depends(get_dataset_service),
]


async def _invoke(
    operation: Callable[..., ResultT],
    *args: Any,
    **kwargs: Any,
) -> ResultT:
    try:
        return await asyncio.to_thread(
            operation,
            *args,
            **kwargs,
        )

    except DatasetIntegrityError:
        logger.exception(
            "Dataset integrity check failed: %s",
            operation.__name__,
        )

        raise _http_error(
            500,
            "dataset_integrity_error",
            "The stored dataset failed an integrity check.",
        ) from None

    except FileNotFoundError:
        raise _http_error(
            404,
            "dataset_not_found",
            "Dataset not found.",
        ) from None

    except FileExistsError:
        raise _http_error(
            409,
            "dataset_conflict",
            "Dataset ID conflict. Retry this upload.",
        ) from None

    except ValueError as exc:
        logger.warning(
            "Dataset input rejected; error_type=%s",
            type(exc).__name__,
        )

        raise _http_error(
            400,
            "invalid_dataset",
            "Check the CSV filename and UTF-8 encoding. "
            "Provide 10–100000 rows and 2–100 unique, "
            "nonblank columns with consistent row widths.",
        ) from None

    except OSError:
        logger.exception(
            "Dataset storage operation failed: %s",
            operation.__name__,
        )

        raise _http_error(
            503,
            "dataset_storage_unavailable",
            "Dataset storage is temporarily unavailable.",
        ) from None

    except Exception:
        logger.exception(
            "Unexpected dataset API error: %s",
            operation.__name__,
        )

        raise _http_error(
            500,
            "internal_error",
            "An unexpected server error occurred.",
        ) from None


@router.post(
    "",
    name="autoaudit_upload_dataset",
    response_model=DatasetMetadata,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and validate one original CSV",
)
async def upload_dataset(
    request: Request,
    response: Response,
    service: ServiceDependency,
    file: Annotated[
        UploadFile,
        File(description="UTF-8 CSV, at most 25 MB."),
    ],
) -> dict[str, Any]:
    filename = file.filename or ""

    if not filename.strip().casefold().endswith(".csv"):
        raise _http_error(
            415,
            "csv_required",
            "Select a file with a .csv extension.",
        )

    if (
        file.size is not None
        and file.size > MAX_CSV_BYTES
    ):
        raise _upload_too_large()

    content = await file.read(
        MAX_CSV_BYTES + 1
    )

    if len(content) > MAX_CSV_BYTES:
        raise _upload_too_large()

    result = await _invoke(
        service.save_bytes,
        filename=filename,
        content=content,
    )

    response.headers["Location"] = str(
        request.url_for(
            "autoaudit_get_dataset",
            dataset_id=result["dataset_id"],
        )
    )

    return result


@router.get(
    "",
    name="autoaudit_list_datasets",
    response_model=list[DatasetMetadata],
    summary="List recent dataset uploads",
)
async def list_datasets(
    service: ServiceDependency,
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 50,
) -> list[dict[str, Any]]:
    return await _invoke(
        service.list_datasets,
        limit=limit,
    )


@router.get(
    "/{dataset_id}",
    name="autoaudit_get_dataset",
    response_model=DatasetMetadata,
    summary="Read saved dataset metadata",
)
async def get_dataset(
    dataset_id: UUID,
    service: ServiceDependency,
) -> dict[str, Any]:
    return await _invoke(
        service.get_metadata,
        dataset_id,
    )


@router.get(
    "/{dataset_id}/profile",
    name="autoaudit_get_dataset_profile",
    response_model=dict[str, Any],
    summary="Profile a dataset after verifying its stored bytes",
)
async def get_dataset_profile(
    dataset_id: UUID,
    service: ServiceDependency,
) -> dict[str, Any]:
    return await _invoke(
        service.get_profile,
        dataset_id,
    )