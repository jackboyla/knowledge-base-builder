import pandas as pd
from pydantic import BaseModel, model_validator, field_validator, Field, ValidationInfo, Extra
from typing import List, Dict, Union, Any, Optional, Literal
import instructor
from openai import OpenAI
from langchain_core.documents.base import Document
import os
import time
import duckduckgo_verbose_search
from tqdm import tqdm
from wikidata_search import WikidataSearch, get_all_properties_with_labels
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
import wikidata_search

import validator_utils
import utils
logger = utils.create_logger(__name__)

client = instructor.patch(OpenAI(api_key=os.environ['OPENAI_API_KEY']))
MODEL = os.environ['VALIDATION_MODEL']
logger.info(f"Using Validator model {MODEL}")


class ValidatedTriple(BaseModel, extra='allow'):
    predicted_subject_name: str
    predicted_relation: Union[str, List[str]]
    predicted_object_name: str

    triple_is_valid: Literal[True, False, "Not enough information to say"] = Field(
      ...,
        description="Whether the predicted subject-relation-object triple is generally valid, following the previously-stated approach. " +
                    "If multiple relations are provided, the triple is valid if any of them is valid. " +
                    "Think through the context and the nuances of the terms before providing your answer. " +
                    "If the context does not provide enough information, try to use your common sense."
    )
    reason: str = Field(
        ..., description="The reason why the predicted subject-relation-object triple is or is not valid."
    )



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
                "content": f"Using your vast knowledge of the world and the given context as a reference to help you if necessary, " + 
                            "evaluate the predicted Knowledge Graph triple for its accuracy by considering:\n" +
                            "1. Definitions, relevance, and any cultural or domain-specific nuances of key terms\n" + 
                            "2. Historical and factual validity, including any recent updates or debates around the information\n" + 
                            "3. The validity of synonyms or related terms of the prediction\n" + 
                            "Approach this with a mindset that allows for exploratory analysis and the recognition of uncertainty or multiple valid perspectives. " +
                            "Use this approach to recognize a range of correct answers when nuances and context allow for it." +
                            "If the context does not provide enough information, try to use your common sense. " +
                            "If multiple relations are provided, the triple is valid if any of them are valid. " +
                            f"\nSubject Name: {entity_label}" + 
                            f"\nRelation: {predicted_property_name}" + 
                            f"\nObject Name: {predicted_property_value}" + 
                            f"\n\nContext: {context}"
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
                "content": f"Using your vast knowledge of the world, " +
                        "evaluate the predicted Knowledge Graph triple for its accuracy by considering:\n" +
                        "1. Definitions, relevance, and any cultural or domain-specific nuances of key terms\n" + 
                        "2. Historical and factual validity, including any recent updates or debates around the information\n" + 
                        "3. The validity of synonyms or related terms of the prediction\n" + 
                        "Approach this with a mindset that allows for exploratory analysis and the recognition of uncertainty or multiple valid perspectives. " +
                        "Use this approach to recognize a range of correct answers when nuances and context allow for it." +
                        "If multiple relations are provided, the triple is valid if any of them are valid. " +
                        f"\nSubject Name: {entity_label}" + 
                        f"\nRelation: {predicted_property_name}" + 
                        f"\nObject Name: {predicted_property_value}"
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
        search_query = f"What {subject} {relation} {object}?"
        return search_query

    @model_validator(mode='before')
    def validate(self, context) -> "WebKGValidator":

        self['validated_triples'] = []

        search_tool = duckduckgo_verbose_search.DuckDuckGoVerboseSearch(max_search_results=5)

        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            search_query = WebKGValidator.create_query(subject, relation, object)

            # web_reference = WebKGValidator.get_web_search_results(search_tool, search_query)

            web_results: List[Dict] = search_tool(search_query)
            web_reference = [Document(f"{result['title']} {result['body']}") for result in web_results]
            retriever, store, vectorstore = validator_utils.create_parent_document_retriever(web_reference)

            relevant_chunks = validator_utils.retrieve_relevant_chunks(
                query=f"{triple['subject']} {triple['relation']} {triple['object']}", 
                vectorstore=vectorstore,
                retriever=retriever,
            )
            reference_context = {'relevant_text': relevant_chunks}

            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context=reference_context
            )
            resp.sources = [reference_context]
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
    def get_wikidata(entity_label, wikidata_wrapper) -> str:

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

        search_tool = duckduckgo_verbose_search.DuckDuckGoVerboseSearch(max_search_results=5)
        wrapper = WikidataAPIWrapper()
        wrapper.top_k_results = 1
        wikidata_wrapper = WikidataQueryRun(api_wrapper=wrapper)

        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            wikidata_reference = WikidataKGValidator.get_wikidata(subject, wikidata_wrapper)
            search_query = WebKGValidator.create_query(subject, relation, object)

            # web_reference = WebKGValidator.get_web_search_results(search_tool, search_query)

            web_results: List[Dict] = search_tool(search_query)
            web_reference = [Document(f"{result['title']} {result['body']}") for result in web_results]
            retriever, store, vectorstore = validator_utils.create_parent_document_retriever(web_reference)

            relevant_chunks = validator_utils.retrieve_relevant_chunks(
                query=f"{triple['subject']} {triple['relation']} {triple['object']}", 
                vectorstore=vectorstore,
                retriever=retriever,
            )
            reference_context = {'web_reference': web_reference, 'wikidata_reference': wikidata_reference}

            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context=reference_context
            )
            resp.sources = reference_context
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



class ReferenceKGValidator(BaseModel):
    ''' Validate triples with LLM's inherent knowledge +  given reference KG
    
        reference_knowledge_graph must be List of triples [{'subject': ..., 'relation': ..., 'object': ...}, ]
    '''

    triples: List
    validated_triples: List[ValidatedTriple] = []
    reference_knowledge_graph: List[Dict]

    @model_validator(mode='before')
    def validate(self, context) -> "ReferenceKGValidator":

        self['validated_triples'] = []


        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            relevant_triples = validator_utils.retrieve_relevant_triples(triple['subject'], self['reference_knowledge_graph'])

            reference_context = {'reference knowledge graph': relevant_triples}

            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context=reference_context
            )
            resp.sources = reference_context
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
    


class WikipediaWikidataKGValidator(BaseModel):
    ''' Validate triples with LLM's inherent knowledge +  Wikipedia + Wikidata as context
    
    '''

    triples: List
    validated_triples: List[ValidatedTriple] = []

    @model_validator(mode='before')
    def validate(self, context) -> "WikipediaWikidataKGValidator":

        self['validated_triples'] = []

        wrapper = WikidataAPIWrapper()
        wrapper.top_k_results = 3
        wikidata_wrapper = WikidataQueryRun(api_wrapper=wrapper)


        for triple in tqdm(self['triples']):

            subject, relation, object = triple['subject'], triple['relation'], triple['object']

            wikidata_reference = WikidataKGValidator.get_wikidata(subject, wikidata_wrapper)

            wikidata_ids = wikidata_search.get_wikidata_qids(triple['subject'])
            if len(wikidata_ids) > 0:
                wikipedia_content = wikidata_search.fetch_wikipedia_page_content(wikidata_ids[0]['id'])

            # logger.info(f"Constructing vectorstore for the reference context...")
            reference = [Document(wikipedia_content)]
            retriever, store, vectorstore = validator_utils.create_parent_document_retriever(reference)
            # logger.info(f"Vectorstore built!")

            relevant_chunks = validator_utils.retrieve_relevant_chunks(
                query=f"{triple['subject']} {triple['relation']} {triple['object']}", 
                vectorstore=vectorstore,
                retriever=retriever,
            )
            reference_context = {'relevant_text': relevant_chunks, 'wikidata_reference': wikidata_reference}

            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context=reference_context
            )
            resp.sources = reference_context
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

            relevant_chunks = validator_utils.retrieve_relevant_chunks(
                query=f"{triple['subject']} {triple['relation']} {triple['object']}", 
                vectorstore=vectorstore,
                retriever=retriever,
            )
            reference_context = {'relevant_text': relevant_chunks}


            # EVALUATE ONE PROPERTY
            resp = validate_statement_with_context(
                entity_label=subject, 
                predicted_property_name=relation, 
                predicted_property_value=object, 
                context=reference_context
            )
            resp.sources = reference_context
            resp.candidate_triple = triple

            self['validated_triples'].append(resp)

        return self
    
