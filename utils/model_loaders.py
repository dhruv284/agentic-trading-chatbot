import os
from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config_loader import load_config


class ModelLoader:
    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self._validate_env()

        self.embeddings = None

    def _validate_env(self):
        required_vars = [
            "GEMINI_API_KEY",
            "JINA_API_KEY"
        ]

        missing = [var for var in required_vars if not os.getenv(var)]

        if missing:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing)}"
            )

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")

    def load_embeddings(self):
        if self.embeddings is None:
            self.embeddings = JinaEmbeddings(
                jina_api_key=self.jina_api_key,
                model_name="jina-embeddings-v3"
            )

        return self.embeddings

    def load_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-3.7-flash",
            google_api_key=self.gemini_api_key,
            temperature=0
        )