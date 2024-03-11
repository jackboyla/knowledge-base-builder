from abc import ABC, abstractmethod

open_ai_models = {'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'}

# Abstract class for all embedding models
class EmbeddingModel(ABC):
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def embed_documents(self, data):
        pass

# generic sent transformers class
class SentenceTransformersEmbeddings(EmbeddingModel):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.name)

    def embed_documents(self, data):
        return self.model.encode(data).tolist()
    
    def embed_query(self, query):
        return self.model.encode(query).tolist()


# A factory to get the appropriate embedding model
class EmbeddingModelFactory:
    @staticmethod
    def get_embedding_model(model_name):
        if model_name in open_ai_models:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name)
        else:
            model = SentenceTransformersEmbeddings(model_name)
            model.load_model()
            return model

