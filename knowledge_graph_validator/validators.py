import pandas as pd
from pydantic import BaseModel, model_validator, field_validator, Field, ValidationInfo, Extra
from typing import List, Dict, Union, Any, Optional, Literal
import instructor
from openai import OpenAI
from langchain_core.documents.base import Document
import os
import time
from duckduckgo_search import DDGS
from tqdm import tqdm
from wikidata_search import WikidataSearch, get_all_properties_with_labels
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

import validator_utils
import utils
logger = utils.create_logger(__name__)

client = instructor.patch(OpenAI(api_key=os.environ['OPENAI_API_KEY']))
MODEL = os.environ['VALIDATION_MODEL']
logger.info(f"Using Validator model {MODEL}")


def validate_reasoning(values):
    chain_of_thought = values["chain_of_thought"]
    answer = values["answer"]
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a validator. Determine if the value is valid for the statement. If it is not, explain why.",
            },
            {
                "role": "user",
                "content": f"Verify that `{answer}` follows the chain of thought: {chain_of_thought}",
            },
        ],
        # this comes from client = instructor.patch(OpenAI())
        response_model=Validation,
    )
    if not resp.is_valid:
        raise ValueError(resp.error_message)
    return values


class ValidatedTriple(BaseModel, extra='allow'):
    predicted_subject_name: str
    predicted_relation: Union[str, List[str]]
    predicted_object_name: Any

    triple_is_valid: Literal[True, False, "Not enough information to say"] = Field(
      ...,
        description="Whether the predicted subject-relation-object triple is generally valid, according to the previously stated rules. " +
                    "If multiple relations are provided, the triple is valid if any of them is valid. " +
                    "Think through the context and the nuances of the terms before providing your answer. " +
                    "If the context does not provide enough information, try to use your common sense."
    )
    reason: str = Field(
        None, description="The reason why the predicted subject-relation-object triple is or is not valid."
    )

    # @model_validator(mode='after')
    # def assert_correct_validation(self, info: ValidationInfo):
    #     if len(self.validated_triples) != len(self.triples):
    #         raise ValueError(
    #             "Number of properties validated does not match number of properties in the prediction knowledge base. " +
    #             f"Number of properties validated: {len(self.validated_triples)}, " +
    #             f"Number of properties in the text: {len(self.triples)}"
    #             )
    #     return self
    



@staticmethod
def validate_statement_with_context(entity_label, predicted_property_name, predicted_property_value, context):
    '''Validate a statement about an entity

    a statement is a triple: entity_label --> predicted_property_name --> predicted_property_value
                            e.g Donald Trump --> wife --> Ivanka Trump
    
    '''
    resp: ValidatedTriple = client.chat.completions.create(
        response_model=ValidatedTriple,
        messages=[
            {
                "role": "user",
                "content": f"Using your knowledge of the world and the given context as a reference, " + 
                            "evaluate the predicted triple for its accuracy by considering: " + 
                            "1. Definitions, relevance, and any cultural or domain-specific nuances of key terms, " + 
                            "2. Historical and factual validity, including any recent updates or debates around the information, " + 
                            "3. The appropriateness of synonyms or related terms, and whether they enrich or detract from the triple's accuracy, " + 
                            "4. The implications and nuanced meanings of the terms within the given context, " + 
                            "5. The contextual relevance and how well the triple integrates with the provided context. " + 
                            "Acknowledge a range of correct answers where appropriate, especially when nuances and context allow for multiple interpretations. Evaluate the strength and appropriateness of each relation if multiple are suggested. " +
                            "Approach this with a mindset that allows for exploratory analysis and the recognition of uncertainty or multiple valid perspectives. " +
                            f"\nSubject Name: {entity_label}" + 
                            f"\nPredicted Relation: {predicted_property_name}" + 
                            f"\nPredicted Object Name: {predicted_property_value}" + 
                            f"\n\nContext: {context}" + 
                            "Use this approach to recognize a range of correct answers when nuances and context allow for it." +
                            "If the context does not provide enough information, try to use your common sense."
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
    resp: ValidatedTriple = client.chat.completions.create(
        response_model=ValidatedTriple,
        messages=[
            {
                "role": "user",
                "content": f"Using your knowledge of the world, " +
                        "evaluate the predicted triple for its accuracy by considering: " +
                        "1. Definitions and relevance of key terms, " +
                        "2. Historical and factual validity, " +
                        "3. Synonyms or related terms appropriateness (i.e, is the prediction 'close enough'), " +
                        "4. Nuances and implications of the terms. " +
                        "5. Any facts you kniw about the entity. " +
                        "Acknowledge a range of correct answers where appropriate. " +
                        "If multiple relations are provided please consider them all individually." +
                        f"\nSubject Name: {entity_label}" +
                        f"\nPredicted Relation: {predicted_property_name}" +
                        f"\nPredicted Object Name: {predicted_property_value}" +
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
    validated_triples: List[ValidatedTriple] = []


    @staticmethod
    def get_web_search_results(search_tool, query, max_retries=6, sleep_interval=10):
        '''Attempt to fetch web search results with retry logic.'''
        for attempt in range(1, max_retries + 1):
            try:
                # Attempt to perform the search
                hits = search_tool.text(query, max_results=5)
                return [h for h in hits]
            except Exception as e:
                # Print or log the error and retry after waiting
                logger.info(f"Attempt {attempt} failed with error: {e}")
                if attempt < max_retries:
                    print("Retrying...")
                    time.sleep(sleep_interval)
                else:
                    # All attempts failed; re-raise the last exception
                    logger.error(f"Failed to get web search results for {query} after {max_retries} attempts")
                    return [""]


    @staticmethod
    def create_query(subject, relation, object):
        '''Create a query for the web search engine'''
        # subject = " ".join(word.capitalize() for word in subject.split("_"))
        # relation = " ".join(relation.split("_"))
        search_query = f"What {subject} {relation} {object}?"
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
            resp = validate_statement_with_context(
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
    validated_triples: List[ValidatedTriple] = []


    @staticmethod
    def get_wikidata(entity_label, wikidata_wrapper):

        q = entity_label
        q = " ".join(q.split('_'))
        try:
            return wikidata_wrapper.run(q)  # a string of the wikidata page
        except Exception as e:
            logger.error(f"Could not get wikidata page for {q} due to {e}")
            return ""

    
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
            resp.sources = [wikidata_reference]
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


class WorldKnowledgeKGValidator(BaseModel):
    ''' Validate triples with LLM's inherent knowledge'''

    triples: List
    validated_triples: List[ValidatedTriple] = []

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
    


class WikidataWebKGValidator(BaseModel):
    ''' Validate triples with LLM's inherent knowledge +  wikidata + web search results'''

    triples: List
    validated_triples: List[ValidatedTriple] = []


    @model_validator(mode='before')
    def validate(self, context) -> "WikidataWebKGValidator":

        self['validated_triples'] = []

        search_tool = DDGS()
        wrapper = WikidataAPIWrapper()
        wrapper.top_k_results = 1
        wikidata_wrapper = WikidataQueryRun(api_wrapper=wrapper)

        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            search_query = WebKGValidator.create_query(subject, relation, object)
            web_reference = WebKGValidator.get_web_search_results(search_tool, search_query)
            wikidata_reference = WikidataKGValidator.get_wikidata(subject, wikidata_wrapper)

            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context={'web_reference': web_reference, 'wikidata_reference': wikidata_reference}
            )
            resp.sources = {'web_reference': web_reference, 'wikidata_reference': wikidata_reference}
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



class TextContextKGValidator(BaseModel):
    ''' Validate triples with LLM's inherent knowledge + given textual context '''

    triples: List
    validated_triples: List[ValidatedTriple] = []



    @model_validator(mode='before')
    def validate(self, context) -> "TextContextKGValidator":

        # create the document store
        docs = [Document(d) for d in self['documents']]
        retriever, store, vectorstore = validator_utils.create_parent_document_retriever(docs)

        self['validated_triples'] = []

        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            relevant_chunk = validator_utils.retrieve_relevant_chunk(
                entity_name=self['entity_label'],
                property_name=subject, 
                vectorstore=vectorstore,
                retriever=retriever
            )


            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context=relevant_chunk
            )
            resp.sources = [relevant_chunk]
            resp.candidate_triple = triple

            self['validated_triples'].append(resp)

        return self
    
