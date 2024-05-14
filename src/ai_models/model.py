"""Speech to text model initialization file"""
from io import BytesIO
import os
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub.utils._validators import HFValidationError

from ai_models.speech2text_interface import Speech2TextInterface
from utils.features_extractor import load_audio, split_audio
from env import USE_CUDA, MODEL_NAME


class Speech2text(Speech2TextInterface):
    """ Speech to text model initialization file."""

    def __init__(self,
            device = None,
            model_name: str = MODEL_NAME,
            language: str = "russian"
        ):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.path_to_model = "./ai_models/weights/"
        self.torch_dtype = torch.float32

        if USE_CUDA is True and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if self.device == "cuda:0":
            self.torch_dtype = torch.float16
        
        self.load_weigths(self.path_to_model)

        # move to device
        self.model.to(self.device)

    def load_weigths(self, path: str):
        """ Download the model weights."""
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                force_download=True
            )

            self.processor = AutoProcessor.from_pretrained(
                path,
                language=self.language,
                task="transcribe",
                force_download=True
            )
        # if we didnt find the model, we try to download it
        except (HFValidationError, OSError):

            # load the model
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                force_download=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                language=self.language,
                force_download=True
            )

            #  save the model
            self.model.save_pretrained(path)
            self.processor.save_pretrained(path)

    def __call__(self, audio: BytesIO | str) -> str:
        """ Get model output from the pipeline.

        Args:
            file_path (UploadFile | str): uploaded file or path to the wav file.

        Returns:
            str: model output.
        """
        inputs = load_audio(audio)
        inputs_chunks = split_audio(inputs, sample_rate=16000)

        transcriptions = []

        for chunk in inputs_chunks:
            # load the vois from path with 16gHz
            input_features = self.processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            input_features = input_features.to(self.device, dtype=self.torch_dtype)

            # get decoder for our language
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=self.language,
                task="transcribe"
            )

            # model inference
            pred_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

            # decode the transcription
            transcription = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            input_features.to("cpu")

            transcriptions.append(transcription)
        return "".join(transcriptions)

    def __str__(self) -> str:
        return self.model_name
    
    def get_config(self):
        """Return information of the model."""
        result = {
            "model" : self.model_name,
            "dtype" : str(self.torch_dtype),
            "device" : self.device,
            "language" : self.language
        }
        return result
