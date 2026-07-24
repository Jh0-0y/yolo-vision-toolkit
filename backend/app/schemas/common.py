"""Shared response DTOs."""

from pydantic import BaseModel


class OkResponse(BaseModel):
    ok: bool = True
