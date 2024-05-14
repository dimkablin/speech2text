"""Main FastAPI entry point."""
import logging
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import gradio as gr

from api.app.middleware import BackendMiddleware
from api.app.endpoint import router
from api.gradio_app import iface


# LOGGING CONFIG SETTING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logging.info("Running speech2text models.")


# APP CONFIG SET
app = FastAPI(
    title="Backend API",
    docs_url="/docs",
    openapi_url="/openapi.json",
    openapi_tags=[{
        "name": "Backend API",
        "description": "API for speech2text models factory method."
    }]
)


# MIDDLEWARE CONFIG SET
app.add_middleware(BackendMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(router, tags=["ai_models"])
app.mount("/docs", StaticFiles(directory="../docs"), name="docs")
app = gr.mount_gradio_app(app, iface, path="/")
