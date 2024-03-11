'''
Utility functions for KG Validators
'''

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


CHARS_PER_TOKEN = 4


@staticmethod
def create_parent_document_retriever(docs: List[Document], embedding_function):
    # https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=embedding_function
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        # parent_splitter=parent_splitter,
    )
    retriever.add_documents(docs, ids=None)

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
def retrieve_fuzzy_triples(query, vectorstore, retriever, reference_kg: List[Dict], top_k=5) -> List[Dict]:
    pass

@staticmethod
def retrieve_relevant_triples(query, reference_kg: List[Dict]) -> List[Dict]:

    relevant_triples = []
    for triple in reference_kg:
        if query in triple['subject'] or query in triple['relation'] or query in triple['object']:
            relevant_triples.append(triple)
    return relevant_triples

@staticmethod
def calc_num_tokens(inp) -> int:
    return len(str(inp)) // CHARS_PER_TOKEN


@staticmethod
def truncate_tokens(inp, max_tokens=15_000) -> str:
    return str(inp)[ : max_tokens * CHARS_PER_TOKEN]


@staticmethod
def retrieve_relevant_chunks(query, vectorstore, retriever, num_chunks=3) -> List:
    '''Fetch the most similar chunk to the given query'''
    relevant_chunks = []
    sub_docs = vectorstore.similarity_search(query)

    # if there is a parent doc chunk, use that, 
    #otherwise use the sub doc chunk
    # parent_chunk = retriever.get_relevant_documents(query)
    # if len(parent_chunk) > 0:
    #     relevant_chunk = parent_chunk[0].page_content
    # else:
    #     relevant_chunk = sub_docs[0].page_content
    for i, chunk in enumerate(sub_docs):
        if i < num_chunks:
            relevant_chunks.append(chunk.page_content)
    return relevant_chunks