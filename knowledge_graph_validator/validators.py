import pandas as pd
from pydantic import BaseModel, model_validator, field_validator, Field, ValidationInfo, Extra
from typing import List, Dict, Union, Any, Optional, Literal
import instructor
from openai import OpenAI
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
import os
from duckduckgo_search import DDGS
from tqdm import tqdm


client = instructor.patch(OpenAI(api_key=os.environ['OPENAI_API_KEY']))
MODEL = "gpt-3.5-turbo-0125"

class ValidatedProperty(BaseModel, extra='allow'):
    entity_label: str
    property_name: str
    property_value: Any

    property_is_valid: Literal[True, False, "Not enough information to say"] = Field(
      ...,
        description="Whether the property is generally valid, judged against " +
                    "the given context.",
    )
    is_valid_reason: Optional[str] = Field(
        None, description="The reason why the property is valid if it is indeed valid."
    )
    error_message: Optional[str] = Field(
        None, description="The error message if either property_name and/or property_value is not valid."
    )


class WebKGValidator(BaseModel):

    triples: List
    validated_triples: List[ValidatedProperty] = []


    @staticmethod
    def get_web_search_results(search_tool, query):
        hits = search_tool.text(query, max_results=5)
        return [h for h in hits]

    @staticmethod
    def create_parent_document_retriever(docs: List[Document]):
        # https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever

        # This text splitter is used to create the child documents
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        # The vectorstore to use to index the child chunks
        vectorstore = Chroma(
            collection_name="full_documents", embedding_function=OpenAIEmbeddings()
        )
        # The storage layer for the parent documents
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            # parent_splitter=parent_splitter,
        )
        retriever.add_documents(docs, ids=None) # add entity doc(s)

        # list(store.yield_keys())   # see how many chunks it's created

        return retriever, store, vectorstore

    @staticmethod
    def retrieve_relevant_property(entity_name, property_name, vectorstore, retriever):
        '''Fetch the most similar chunk to predicted property name'''

        query = f"{property_name}"

        sub_docs = vectorstore.similarity_search(query)

        relevant_property = sub_docs[0].page_content
        return relevant_property

    @staticmethod
    def validate_statement_with_context(entity_label, predicted_property_name, predicted_property_value, context):
        '''Validate a statement about an entity

        a statement is a triple: entity_label --> predicted_property_name --> predicted_property_value
                             e.g Donald Trump --> wife --> Ivanka Trump
        
        '''
        resp: ValidatedProperty = client.chat.completions.create(
            response_model=ValidatedProperty,
            messages=[
                {
                    "role": "user",
                    "content": f"Using the given context as a reference, " +
                            "evaluate the predicted property for its accuracy by considering: " +
                            "1. Definitions and relevance of key terms, " +
                            "2. Historical and factual validity, " +
                            "3. Synonyms or related terms appropriateness, " +
                            "4. Nuances and implications of the terms. " +
                            "Acknowledge a range of correct answers where appropriate. " +
                            f"\nEntity Label: {entity_label}" +
                            f"\nPredicted Property Name: {predicted_property_name}" +
                            f"\nPredicted Property Value: {predicted_property_value}" +
                            f"\n\nContext: {context}" +
                            "Use this approach to recognize a range of correct answers when nuances and context allow for it."
                }
            ],
            max_retries=2,
            temperature=0,
            model=MODEL,
        )
        return resp


    @staticmethod
    def create_query(subject, relation, object=None):
        '''Create a query for the web search engine'''
        # subject = " ".join(word.capitalize() for word in subject.split("_"))
        # relation = " ".join(relation.split("_"))
        search_query = f"What {subject} {relation}?"
        return search_query

    @model_validator(mode='before')
    def validate(self, context) -> "WebKGValidator":

        self['validated_triples'] = []

        search_tool = DDGS()

        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            search_query = WebKGValidator.create_query(subject, relation, object)

            web_reference = WebKGValidator.get_web_search_results(search_tool, search_query)

            # EVALUATE ONE PROPERTY
            resp = WebKGValidator.validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context=web_reference
            )
            resp.sources = [web_reference]
            resp.candidate_triple = triple

            self['validated_triples'].append(resp)
        return self


    @model_validator(mode='after')
    def assert_all_properties_validated(self, info: ValidationInfo):
        if len(self.validated_triples) != len(self.triples):
            raise ValueError(
                "Number of properties validated does not match number of properties in the prediction knowledge base. " +
                f"Number of properties validated: {len(self.validated_triples)}, " +
                f"Number of properties in the text: {len(self.triples)}"
                )
        return self

