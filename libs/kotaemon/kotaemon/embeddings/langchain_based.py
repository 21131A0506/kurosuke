from typing import Optional

from kotaemon.base import DocumentWithEmbedding, Param

from .base import BaseEmbeddings


class LCEmbeddingMixin:
    def _get_lc_class(self):
        raise NotImplementedError(
            "Please return the relevant Langchain class in in _get_lc_class"
        )

    def __init__(self, **params):
        self._lc_class = self._get_lc_class()
        self._obj = self._lc_class(**params)
        self._kwargs: dict = params

        super().__init__()

    def run(self, text):
        input_docs = self.prepare_input(text)
        input_ = [doc.text for doc in input_docs]

        embeddings = self._obj.embed_documents(input_)

        return [
            DocumentWithEmbedding(content=doc, embedding=each_embedding)
            for doc, each_embedding in zip(input_docs, embeddings)
        ]

    def __repr__(self):
        kwargs = []
        for key, value_obj in self._kwargs.items():
            value = repr(value_obj)
            kwargs.append(f"{key}={value}")
        kwargs_repr = ", ".join(kwargs)
        return f"{self.__class__.__name__}({kwargs_repr})"

    def __str__(self):
        kwargs = []
        for key, value_obj in self._kwargs.items():
            value = str(value_obj)
            if len(value) > 20:
                value = f"{value[:15]}..."
            kwargs.append(f"{key}={value}")
        kwargs_repr = ", ".join(kwargs)
        return f"{self.__class__.__name__}({kwargs_repr})"

    def __setattr__(self, name, value):
        if name == "_lc_class":
            return super().__setattr__(name, value)

        if name in self._lc_class.__fields__:
            self._kwargs[name] = value
            self._obj = self._lc_class(**self._kwargs)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._kwargs:
            return self._kwargs[name]
        return getattr(self._obj, name)

    def dump(self, *args, **kwargs):
        from theflow.utils.modules import serialize

        params = {key: serialize(value) for key, value in self._kwargs.items()}
        return {
            "__type__": f"{self.__module__}.{self.__class__.__qualname__}",
            **params,
        }

    def specs(self, path: str):
        path = path.strip(".")
        if "." in path:
            raise ValueError("path should not contain '.'")

        if path in self._lc_class.__fields__:
            return {
                "__type__": "theflow.base.ParamAttr",
                "refresh_on_set": True,
                "strict_type": True,
            }

        raise ValueError(f"Invalid param {path}")




class LCGoogleEmbeddings(LCEmbeddingMixin, BaseEmbeddings):
    """Wrapper around Langchain's Google GenAI embedding, focusing on key parameters"""

    google_api_key: str = Param(
        help="API key (https://aistudio.google.com/app/apikey)",
        default=None,
        required=True,
    )
    model: str = Param(
        help="Model name to use (https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding-and-embedding)",  # noqa
        default="models/text-embedding-004",
        required=True,
    )

    def __init__(
        self,
        model: str = "models/text-embedding-004",
        google_api_key: Optional[str] = None,
        **params,
    ):
        super().__init__(
            model=model,
            google_api_key=google_api_key,
            **params,
        )

    def _get_lc_class(self):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError("Please install langchain-google-genai")

        return GoogleGenerativeAIEmbeddings
