"""Project-class request DTOs."""

from pydantic import BaseModel


class ClassIn(BaseModel):
    name: str
