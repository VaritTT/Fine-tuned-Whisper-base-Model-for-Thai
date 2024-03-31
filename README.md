[📔 Jupyter Notebook](https://github.com/VaritTT/Fine-tuned-Whisper-base-Model-for-Thai/blob/main/fine_tuned_Whisper_base_model_by_Varit_Tubtim.ipynb) | [🤗 Huggingface Demo](https://huggingface.co/Varit/whisper-base-th-project-final)

<a target="_blank" href="https://colab.research.google.com/github/VaritTT/Fine-tuned-Whisper-base-Model-for-Thai/blob/main/fine_tuned_Whisper_base_model_by_Varit_Tubtim.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
## Fine-tuned Whisper-base Model for Thai Demo
Speech Recognition for Thai using Whisper-base Model from OpenAI.
Dataset from Common Voice Corpus 6.1 in https://huggingface.co/datasets/mozilla-foundation/common_voice_6_1


## Setup
```bash
pip install gradio
```

## Usage
```py
from transformers import pipeline
import gradio as gr

pipe = pipeline(model="Varit/whisper-base-th-project-final")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs="text",
    title="Whisper base Thai",
    description="Realtime demo for Thai speech recognition using a fine-tuned Whisper base model.",
)

iface.launch()
```
