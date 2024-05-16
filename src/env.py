""" Script to load environments variables """

import os

# importing environment variables from .env file
USE_CUDA = eval(os.getenv("USE_CUDA", default="True"))
MODEL_NAME = os.getenv("MODEL_NAME", default="openai/whisper-small")
