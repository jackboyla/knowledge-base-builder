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