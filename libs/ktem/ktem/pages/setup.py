import json

import gradio as gr
import requests
from decouple import config
from ktem.app import BasePage
from ktem.embeddings.manager import embedding_models_manager as embeddings
from ktem.llms.manager import llms
from theflow.settings import settings as flowsettings

DEMO_MESSAGE = (
    "This is a public space. Please use the "
    '"Duplicate Space" function on the top right '
    "corner to setup your own space."
)

class SetupPage(BasePage):

    public_events = ["onFirstSetupComplete"]

    def __init__(self, app):
        self._app = app
        self.on_building_ui()

    def on_building_ui(self):
        gr.Markdown(f"# Welcome to {self._app.app_name} first setup!")
        self.radio_model = gr.Radio(
            [
                ("Google API (*free registration*)", "google"),
            ],
            label="Select your model provider",
            value="google",
            interactive=True,
        )

        with gr.Column(visible=True) as self.google_option:
            gr.Markdown(
                (
                    "#### Google API Key\n\n"
                    "(register your free API key "
                    "at https://aistudio.google.com/app/apikey)"
                )
            )
            self.google_api_key = gr.Textbox(
                show_label=False, placeholder="Google API Key"
            )

        self.setup_log = gr.HTML(show_label=False)

        with gr.Row():
            self.btn_finish = gr.Button("Proceed", variant="primary")
            self.btn_skip = gr.Button("Skip setup", variant="stop")

    def on_register_events(self):
        onFirstSetupComplete = gr.on(
            triggers=[self.btn_finish.click],
            fn=self.update_model,
            inputs=[self.google_api_key, self.radio_model],
            outputs=[self.setup_log],
            show_progress="hidden",
        )
        
        onSkipSetup = gr.on(
            triggers=[self.btn_skip.click],
            fn=lambda: None,
            inputs=[],
            show_progress="hidden",
            outputs=[],
        )

        for event in self._app.get_event("onFirstSetupComplete"):
            onFirstSetupComplete = onFirstSetupComplete.success(**event)

    def update_model(self, google_api_key, radio_model_value):
        log_content = ""
        if radio_model_value == "google" and google_api_key:
            llms.update(
                name="google",
                spec={
                    "__type__": "kotaemon.llms.chats.LCGeminiChat",
                    "model_name": "gemini-1.5-flash",
                    "api_key": google_api_key,
                },
                default=True,
            )
            embeddings.update(
                name="google",
                spec={
                    "__type__": "kotaemon.embeddings.LCGoogleEmbeddings",
                    "model": "models/text-embedding-004",
                    "google_api_key": google_api_key,
                },
                default=True,
            )
        
        log_content += "- Testing LLM model: Google API\n"
        yield log_content

        try:
            llm_output = llms.get("google")("Hi")
            log_content += "- Connection success.\n"
        except Exception as e:
            log_content += f"- Connection failed: {str(e)}\n"
            yield log_content
            raise gr.Error("Setup failed. Check your API key and connection.")

        yield log_content
        gr.Info("Setup completed successfully!")

    def update_default_settings(self, radio_model_value, default_settings):
        default_settings["index.options.1.reranking_llm"] = radio_model_value
        return default_settings
