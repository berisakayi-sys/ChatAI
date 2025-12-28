import gradio as gr
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

def chat(message):
    output = generator(
        message,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )
    return output[0]["generated_text"]

gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="ChatAI",
    description="A free AI chatbot running on open-source models."
).launch()