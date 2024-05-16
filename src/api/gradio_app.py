import gradio as gr
import soundfile as sf
from io import BytesIO
from api.app.endpoint import speech2text


async def speech_to_text(audio):
    sr, samples = audio

    # конвернтнуть из нампи в байты форматом PCM_16 в WAV файл
    # чтобы использовать существующий код
    bytes_io = BytesIO()
    sf.write(bytes_io, samples, sr, subtype='PCM_16', format='WAV')
    bytes_io.seek(0)
    
    result = speech2text(bytes_io)
    return result


input_audio = gr.Audio(
    sources=["microphone"],
    waveform_options=gr.WaveformOptions(
        waveform_color="#00B473",
        waveform_progress_color="#00B473",
        skip_length=2,
        show_controls=False,
    ),
    interactive=True,
    show_download_button=True
)

with gr.Blocks() as iface:
    gr.Markdown(
    """
    # Speech2text model
    """
    )

    gr.JSON(speech2text.get_config())

    gr.Interface(
    fn=speech_to_text,
    inputs=input_audio,
    outputs="textbox", 
    allow_flagging=False
)
