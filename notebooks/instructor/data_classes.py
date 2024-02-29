from pydantic import BaseModel, model_validator, field_validator, Field, ValidationInfo
from typing import List, Dict, Union, Any, Optional, Literal
import instructor


class ValidatedProperty(BaseModel):
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


class KnowledgeGraph(BaseModel):
    entity_label: str
    properties: Dict[str, Any]
