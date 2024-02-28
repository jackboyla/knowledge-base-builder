from utils import read_jsonl, save_jsonl
import pandas as pd
from pydantic import BaseModel, model_validator, field_validator, Field, ValidationInfo
from typing import List, Dict, Union, Any, Optional
import instructor
from openai import OpenAI
import os
import json

client = instructor.patch(OpenAI(api_key=os.environ['OPENAI_API_KEY']))
MODEL = "gpt-3.5-turbo-0125"



#######################################

# v0.1

#######################################

class ValidatedProperty(BaseModel):
    property_name: str
    property_value: Union[List[str], str, int, Dict[str, str]]
    is_valid: bool = Field(
      ...,
        description="""Whether the property value is valid or not, judged 
        against the reference knowledge base.""",
    )
    error_message: Optional[str] = Field(
        None, description="The error message if the value is not valid."
    )
    matching_reference_property: Optional[str] = Field(
        ...,
        description="""The corresponding property in the reference knowledge base which 
        this predicted property correctly answers."""
    )


class ValidatedKnowledgeBase(BaseModel):
    entity_label: str
    properties: List[ValidatedProperty] = Field(
        ...,
        description="A list of validations about whether the property is valid given the reference knowledge base.",
    )

    @field_validator('properties')
    @classmethod
    def assert_all_properties_validated(cls, validation_properties: List[ValidatedProperty], info: ValidationInfo):
        existing_pred_properties = info.context.get("existing_pred_properties")
        if len(validation_properties) != len(existing_pred_properties):
            raise ValueError(
                f"""Number of properties validated does not match number of properties in the prediction knowledge base. 
                Number of properties validated: {len(validation_properties)}, 
                Number of properties in the text: {len(existing_pred_properties)}"""
                )
        return validation_properties

    @field_validator('properties')
    @classmethod
    def assert_matching_ref_in_ref_kb(cls, validation_properties: List[ValidatedProperty], info: ValidationInfo):
        '''
        Make sure that the reference property that is linked to the prediction 
        actually exists in the reference knowledge base.
        '''
        existing_ref_properties = info.context.get("existing_ref_properties")
        for prop in validation_properties:
            if prop.is_valid:
                if prop.matching_reference_property not in existing_ref_properties:
                    raise ValueError(
                        f"""The predicted property name {prop.property_name} was marked valid but the matching_reference_property {prop.matching_reference_property} does not exist in the reference knowledge base."""
                    )
        return validation_properties


class KnowledgeBase(BaseModel):
    entity_label: str
    properties: Dict[str, Any]


class EvaluationKB(BaseModel):
    predicted_knowledge_base: KnowledgeBase = Field(
        ...,
        description="The predicted knowledge base that must be evaluated against the reference."
    )
    reference_knowledge_base: KnowledgeBase = Field(
        ..., 
        description="The reference knowledge base used for evaluating prediction."
    )
    validated_kb: ValidatedKnowledgeBase = None


    @model_validator(mode="after")
    def eval_kb(self, context: str):

        existing_pred_properties = list(self.predicted_knowledge_base.properties.keys())
        existing_ref_properties = list(self.reference_knowledge_base.properties.keys())

        resp = client.chat.completions.create(
            response_model=ValidatedKnowledgeBase,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a knowledge base evaluator.""",
                },
                {
                    "role": "user",
                    "content": f"""Using your knowledge of the world and 
                    the given reference knowledge base, evaluate the given predicted knowledge base.
                    Please note that we are interested in the property names that are predicted as well as the corresponding property values. 
                    \nPredicted Knowledge Base: {self.predicted_knowledge_base}
                    \n\nReference Knowledge Base: {self.reference_knowledge_base}""",
                }
            ],
            validation_context={
                "existing_ref_properties": existing_ref_properties,
                "existing_pred_properties": existing_pred_properties,
            },
            max_retries=4,
            model=MODEL,
        )
        self.validated_kb = resp
            
        return self
    





#######################################

# v0.2

#######################################




class ValidatedProperty(BaseModel):
    property_name: str
    property_value: Union[List[str], str, int, Dict[str, str]]
    property_name_is_valid: bool = Field(
      ...,
        description="""Whether the property name is valid or not, judged against the reference knowledge base.""",
    )
    property_value_is_valid: bool = Field(
      ...,
        description="""Whether the property value is valid or not, judged against the reference knowledge base.""",
    )
    error_message: Optional[str] = Field(
        None, description="The error message if either property name and/or property value is not valid."
    )
    matching_reference_property: Optional[str] = Field(
        ...,
        description="""The corresponding property name in the reference knowledge base which this predicted property_value correctly answers."""
    )


class ValidatedKnowledgeBase(BaseModel):
    entity_label: str
    properties: List[ValidatedProperty] = Field(
        ...,
        description="A list of validations about whether the property is valid given the reference knowledge base.",
    )

    @field_validator('properties')
    @classmethod
    def assert_all_properties_validated(cls, validation_properties: List[ValidatedProperty], info: ValidationInfo):
        existing_pred_properties = info.context.get("existing_pred_properties")
        if len(validation_properties) != len(existing_pred_properties):
            raise ValueError(
                "Number of properties validated does not match number of properties in the prediction knowledge base. " +
                "Number of properties validated: {len(validation_properties)}, " +
                f"Number of properties in the text: {len(existing_pred_properties)}"
                )
        return validation_properties

    @field_validator('properties')
    @classmethod
    def assert_matching_ref_in_ref_kb(cls, validation_properties: List[ValidatedProperty], info: ValidationInfo):
        '''
        Make sure that the reference property that is linked to the prediction 
        actually exists in the reference knowledge base.
        '''
        existing_ref_properties = info.context.get("existing_ref_properties")
        for prop in validation_properties:
            if prop.property_name_is_valid:
                if prop.matching_reference_property not in existing_ref_properties:
                    raise ValueError(
                        f"The predicted property name {prop.property_name} was marked valid but the matching_reference_property " +
                        f"{prop.matching_reference_property} does not exist in the reference knowledge base."
                    )
        return validation_properties


class KnowledgeBase(BaseModel):
    entity_label: str
    properties: Dict[str, Any]


class EvaluationKB(BaseModel):
    predicted_knowledge_base: KnowledgeBase = Field(
        ...,
        description="The predicted knowledge base that must be evaluated against the reference."
    )
    reference_knowledge_base: KnowledgeBase = Field(
        ..., 
        description="The reference knowledge base used for evaluating prediction."
    )
    validated_knowledge_base: ValidatedKnowledgeBase = None


    @model_validator(mode="after")
    def eval_kb(self, context: str):

        existing_pred_properties = list(self.predicted_knowledge_base.properties.keys())
        existing_ref_properties = list(self.reference_knowledge_base.properties.keys())

        resp = client.chat.completions.create(
            response_model=ValidatedKnowledgeBase,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a knowledge base evaluator.""",
                },
                {
                    "role": "user",
                    "content": # Please use this style of formatting prompt, because one f-string will include indents
                        "Using your knowledge of the world and " +
                        "the given reference knowledge base, evaluate the given predicted knowledge base. " +
                        "Please note that we are interested in evaluating the property names that are " +
                        "predicted as well as the corresponding property values. " +
                        f"\nPredicted Knowledge Base: {self.predicted_knowledge_base}" +
                        f"\n\nReference Knowledge Base: {self.reference_knowledge_base}"
                },
            ],
            validation_context={
                "existing_ref_properties": existing_ref_properties,
                "existing_pred_properties": existing_pred_properties,
            },
            max_retries=6,
            model=MODEL,
        )
        self.validated_knowledge_base = resp
            
        return self
    







#######################################

# v0.3

#######################################




class ValidatedProperty(BaseModel):
    property_name: str
    property_value: Union[List[str], str, int, Dict[str, str]]
    property_name_is_valid: bool = Field(
      ...,
        description="A predicted property name is valid if is semantically close " +
                    "to a property name in the reference knowledge base.",
    )
    property_value_is_valid: bool = Field(
      ...,
        description="Whether the property value is generally valid, judged against the " +
                    "reference knowledge base.",
    )
    error_message: Optional[str] = Field(
        None, description="The error message if either property_name and/or property_value is not valid."
    )
    matching_reference_property: Optional[str] = Field(
        ...,
        description="If the predicted property_name is valid, " +
                    "provide the corresponding property_name in the reference knowledge base."
    )


class ValidatedKnowledgeBase(BaseModel):
    entity_label: str
    properties: List[ValidatedProperty] = Field(
        ...,
        description="A list of validations about whether the property is valid given the reference knowledge base.",
    )

    @field_validator('properties')
    @classmethod
    def assert_all_properties_validated(cls, validation_properties: List[ValidatedProperty], info: ValidationInfo):
        existing_pred_properties = info.context.get("existing_pred_properties")
        if len(validation_properties) != len(existing_pred_properties):
            raise ValueError(
                "Number of properties validated does not match number of properties in the prediction knowledge base. " +
                "Number of properties validated: {len(validation_properties)}, " +
                f"Number of properties in the text: {len(existing_pred_properties)}"
                )
        return validation_properties

    @field_validator('properties')
    @classmethod
    def assert_matching_ref_in_ref_kb(cls, validation_properties: List[ValidatedProperty], info: ValidationInfo):
        '''
        Make sure that the reference property that is linked to the prediction 
        actually exists in the reference knowledge base.
        '''
        existing_ref_properties = info.context.get("existing_ref_properties")
        for prop in validation_properties:
            if prop.property_name_is_valid:
                if prop.matching_reference_property not in existing_ref_properties:
                    raise ValueError(
                        f"The predicted property name {prop.property_name} was marked valid but the matching_reference_property " +
                        f"{prop.matching_reference_property} does not exist in the reference knowledge base."
                    )
        return validation_properties


class KnowledgeBase(BaseModel):
    entity_label: str
    properties: Dict[str, Any]


class EvaluationKB(BaseModel):
    predicted_knowledge_base: KnowledgeBase = Field(
        ...,
        description="The predicted knowledge base that must be evaluated against the reference."
    )
    reference_knowledge_base: KnowledgeBase = Field(
        ..., 
        description="The reference knowledge base used for evaluating prediction."
    )
    validated_knowledge_base: ValidatedKnowledgeBase = None


    @model_validator(mode="after")
    def eval_kb(self, context: str):

        existing_pred_properties = list(self.predicted_knowledge_base.properties.keys())
        existing_ref_properties = list(self.reference_knowledge_base.properties.keys())

        resp = client.chat.completions.create(
            response_model=ValidatedKnowledgeBase,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a knowledge base evaluator.""",
                },
                {
                    "role": "user",
                    "content": # Please use this style of formatting prompt, because one f-string will include indents
                        "I want you to evaluate an open-form knowledge base builder model. The model has predicted a knowledge " +
                        "base about an entity from unstructured text, which I provde below. I also provide the reference knowledge base " +
                        "for this entity, which you must use to evaluate the prediction. Please bear in mind that while property names and values " +
                        "may not exactly match reference properties, they may still be generally correct. " +
                        "With this in mind, it is your job to discern whether the predicted property names and values are valid. " +
                        f"\nPredicted Knowledge Base: {self.predicted_knowledge_base}" +
                        f"\n\nReference Knowledge Base: {self.reference_knowledge_base}"
                },
            ],
            validation_context={
                "existing_ref_properties": existing_ref_properties,
                "existing_pred_properties": existing_pred_properties,
            },
            max_retries=6,
            model=MODEL,
        )
        self.validated_knowledge_base = resp
            
        return self
    

