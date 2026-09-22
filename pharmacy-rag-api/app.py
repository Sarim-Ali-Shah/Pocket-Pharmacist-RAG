import gradio as gr
import uvicorn
from app.main import app

# Friendly status dashboard when viewing the Space on Hugging Face
with gr.Blocks(title="Pharmacy RAG API") as demo:
    gr.Markdown("# 💊 Pharmacy RAG API")
    gr.Markdown("The FastAPI backend is **online and running**.")
    gr.Markdown("- Interactive Swagger Documentation: [/docs](/docs)")
    gr.Markdown("- Alternative ReDoc: [/redoc](/redoc)")
    gr.Markdown("- Health Check: [/books/clinical_pharmacy](/books/clinical_pharmacy)")

# Mount Gradio onto the existing FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
