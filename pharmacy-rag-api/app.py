import gradio as gr
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

import uvicorn
from app.main import app

@spaces.GPU
def check_status(x):
    return "API Online"

# Friendly status dashboard when viewing the Space on Hugging Face
with gr.Blocks(title="Pharmacy RAG API") as demo:
    gr.Markdown("# 💊 Pharmacy RAG API")
    gr.Markdown("The FastAPI backend is **online and running**.")
    gr.Markdown("- Interactive Swagger Documentation: [/docs](/docs)")
    gr.Markdown("- Alternative ReDoc: [/redoc](/redoc)")
    gr.Markdown("- Health Check: [/books/clinical_pharmacy](/books/clinical_pharmacy)")

    btn = gr.Button("Status Check", visible=False)
    out = gr.Textbox(visible=False)
    btn.click(fn=check_status, inputs=out, outputs=out)

# Mount Gradio onto the existing FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
