"""Main FastAPI entry point."""
from io import BytesIO
import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ai_models.model import Speech2text


router = APIRouter()
speech2text = Speech2text()


@router.post("/predict", response_model=str)
async def speech_to_text(audio: UploadFile = File(...)) -> str:
    """Predict function."""
    path = os.path.join("data", audio.filename)

    # read and save wav file
    with open(path, "wb") as f:
        f.write(await audio.read())

    result = speech2text(path)

    # delete file
    os.remove(path)

    return JSONResponse(
        status_code=200,
        content={"result": result}
    )


@router.get("/get-config", response_model=dict)
async def get_model_config() -> dict:
    """Return the config of the model"""
    return speech2text.get_config()
