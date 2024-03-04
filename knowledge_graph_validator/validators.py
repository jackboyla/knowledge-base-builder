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
from wikidata_search import WikidataSearch, get_all_properties_with_labels
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

from validator_utils import (
    create_parent_document_retriever,
    retrieve_relevant_property,
)

client = instructor.patch(OpenAI(api_key=os.environ['OPENAI_API_KEY']))
MODEL = "gpt-3.5-turbo-0125"


class ValidatedProperty(BaseModel, extra='allow'):
    subject_name: str
    relation: str
    object_name: Any

    triple_is_valid: Literal[True, False, "Not enough information to say"] = Field(
      ...,
        description="Whether the subject-relation-object triple is generally valid, according to the previously stated rules.",
    )
    reason: str = Field(
        None, description="The reason why the subject-relation-object triple is or is not valid."
    )
    # is_valid_reason: Optional[str] = Field(
    #     None, description="The reason why the subject-relation-object triple is valid, if it is indeed valid."
    # )
    # error_message: Optional[str] = Field(
    #     None, description="The error message if the subject-relation-object triple is not valid."
    # )

    # @model_validator(mode='after')
    # def assert_decision_is_made(self, info: ValidationInfo):
    #     assert self.is_valid_reason is not None or self.error_message is None, "Fields `is_valid_reason` and `error_message` cannot both be filled."
    #     return self


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
                "role": "system",
                "content": "You are a Knowledge Graph Completion triple evaluator."
            },
            {
                "role": "user",
                "content": f"Using your knowledge of the world and the given context as a reference, " +
                        "evaluate the predicted property for its accuracy by considering: " +
                        "1. Definitions and relevance of key terms, " +
                        "2. Historical and factual validity, " +
                        "3. Synonyms or related terms appropriateness, " +
                        "4. Nuances and implications of the terms. " +
                        "5. Any facts you can glean from the context. " +
                        "Acknowledge a range of correct answers where appropriate. " +
                        f"\nEntity Label: {entity_label}" +
                        f"\nPredicted Property Name: {predicted_property_name}" +
                        f"\nPredicted Property Value: {predicted_property_value}" +
                        f"\n\nContext: {context}" +
                        "Use this approach to recognize a range of correct answers when nuances and context allow for it."
            }
        ],
        max_retries=3,
        temperature=0,
        model=MODEL,
    )
    return resp



@staticmethod
def validate_statement_with_no_context(entity_label, predicted_property_name, predicted_property_value):
    '''Validate a statement about an entity with no context

    a statement is a triple: entity_label --> predicted_property_name --> predicted_property_value
                            e.g Donald Trump --> wife --> Ivanka Trump
    
    '''
    resp: ValidatedProperty = client.chat.completions.create(
        response_model=ValidatedProperty,
        messages=[
            {
                "role": "system",
                "content": "You are a Knowledge Graph Completion triple evaluator."
            },
            {
                "role": "user",
                "content": f"Using your knowledge of the world, " +
                        "evaluate the predicted property for its accuracy by considering: " +
                        "1. Definitions and relevance of key terms, " +
                        "2. Historical and factual validity, " +
                        "3. Synonyms or related terms appropriateness, " +
                        "4. Nuances and implications of the terms. " +
                        "5. Any facts you kniw about the entity. " +
                        "Acknowledge a range of correct answers where appropriate. " +
                        f"\nEntity Label: {entity_label}" +
                        f"\nPredicted Property Name: {predicted_property_name}" +
                        f"\nPredicted Property Value: {predicted_property_value}" +
                        "Use this approach to recognize a range of correct answers when nuances and context allow for it."
            }
        ],
        max_retries=3,
        temperature=0,
        model=MODEL,
    )
    return resp



class WebKGValidator(BaseModel):
    ''' Validate triples with LLM's inherent knowledge + web search results'''

    triples: List
    validated_triples: List[ValidatedProperty] = []


    @staticmethod
    def get_web_search_results(search_tool, query):
        hits = search_tool.text(query, max_results=5)
        return [h for h in hits]


    @staticmethod
    def create_query(subject, relation, object):
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
    def assert_all_triples_validated(self, info: ValidationInfo):
        if len(self.validated_triples) != len(self.triples):
            raise ValueError(
                "Number of properties validated does not match number of properties in the prediction knowledge base. " +
                f"Number of properties validated: {len(self.validated_triples)}, " +
                f"Number of properties in the text: {len(self.triples)}"
                )
        return self




class WikidataKGValidator(BaseModel):
    ''' Validate triples with LLM's inherent knowledge + Wikidata'''

    triples: List
    validated_triples: List[ValidatedProperty] = []


    @staticmethod
    def get_wikidata(entity_label, wikidata_wrapper):

        q = entity_label
        q = " ".join(q.split('_'))
        return wikidata_wrapper.run(q)  # a string of the wikidata page

    
    @staticmethod
    def get_wikidata_neighbors(wikidata_kg) -> Dict:
        one_hop_kg = {}
        for property in wikidata_kg['properties'].keys():
            qid: List = WikidataSearch.search_wikidata(property)
            if len(qid) > 0:
                print(f"Found wikidata id for {property}")
                one_hop_kg[property] = get_all_properties_with_labels(qid[0]['id'])

        return one_hop_kg


    @model_validator(mode='before')
    def validate(self, context) -> "WikidataKGValidator":

        self['validated_triples'] = []

        wrapper = WikidataAPIWrapper()
        wrapper.top_k_results = 1
        wikidata_wrapper = WikidataQueryRun(api_wrapper=wrapper)

        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            wikidata_reference = WikidataKGValidator.get_wikidata(subject, wikidata_wrapper)

            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context=wikidata_reference
            )

            self['validated_triples'].append(resp)

        return self


    @model_validator(mode='after')
    def assert_all_triples_validated(self, info: ValidationInfo):
        if len(self.validated_triples) != len(self.triples):
            raise ValueError(
                "Number of properties validated does not match number of properties in the prediction knowledge base. " +
                f"Number of properties validated: {len(self.validated_triples)}, " +
                f"Number of properties in the text: {len(self.triples)}"
                )
        return self


class WorldKnowledgeKGValidator(BaseModel):
    ''' Validate triples with LLM's inherent knowledge'''

    triples: List
    validated_triples: List[ValidatedProperty] = []

    @model_validator(mode='before')
    def validate(self, context) -> "WorldKnowledgeKGValidator":

        self['validated_triples'] = []

        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_no_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object
            )
            resp.candidate_triple = triple

            self['validated_triples'].append(resp)
        return self
    
    @model_validator(mode='after')
    def assert_all_triples_validated(self, info: ValidationInfo):
        if len(self.validated_triples) != len(self.triples):
            raise ValueError(
                "Number of properties validated does not match number of properties in the prediction knowledge base. " +
                f"Number of properties validated: {len(self.validated_triples)}, " +
                f"Number of properties in the text: {len(self.triples)}"
                )
        return self