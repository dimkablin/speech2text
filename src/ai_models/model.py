"""Speech to text model initialization file"""
from io import BytesIO
import torch

from transformers import (
    WhisperForConditionalGeneration,
    AutoProcessor, 
    AutoModelForSpeechSeq2Seq, 
    pipeline
)

from ai_models.speech2text_interface import Speech2TextInterface
from utils.features_extractor import *
from env import USE_CUDA, MODEL_NAME


class Speech2text(Speech2TextInterface):
    """ Speech to text model initialization file."""

    def __init__(self,
                 device=None,
                 model_name: str = MODEL_NAME,
                 dtype: torch.dtype = torch.float32,
                 language: str = "russian"):
        self.model_name = model_name
        self.dtype = dtype
        self.language = language
        self.sample_rate = 16000

        if USE_CUDA is True and torch.cuda.is_available():
            self.device = "cuda:0"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
        self.device = torch.device(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True
        ).to(device=self.device, dtype=self.dtype)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            return_timestamps=False,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=self.dtype,
            device=self.device,
        )

    def __call__(self, audio: BytesIO | str) -> str:
        """ Get model output from the pipeline.

        Args:
            audio (BytesIO | str): uploaded file or path to the wav file.

        Returns:
            str: model output.
        """
        input_ = load_audio(audio).numpy()
        input_ = reduce_noise(input_, sample_rate=self.sample_rate)

        kwargs = {}
        if isinstance(self.model, WhisperForConditionalGeneration):
            kwargs["generate_kwargs"]={"language": self.language, "task": "transcribe"}

        output = self.pipe(input_, **kwargs)
        return output['text']

    def __str__(self) -> str:
        return self.model_name
    
    def get_config(self):
        """Return information of the model."""
        result = {
            "model": self.model_name,
            "dtype": str(self.model.dtype),
            "device": str(self.model.device),
            "languge": self.language
        }
        return result
