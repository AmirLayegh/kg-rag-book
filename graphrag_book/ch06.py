import dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
from openai import OpenAI
import json
from utils import neo4j_driver

contract_types = [
    "Service Agreement",
    "Licensing Agreement",
    "Non-Disclosure Agreement",
    "Partnership Agreement",
    "Lease Agreement",
]

class Location(BaseModel):
    address: Optional[str] = Field(
        ...,
        description="The address of the location."
    )
    city: Optional[str] = Field(
        ...,
        description="The city of the location."
    )
    state: Optional[str] = Field(
        ...,
        description="The state of the location."
    )
    country: Optional[str] = Field(
        ...,
        description="The country of the location."
    )
    
class Organization(BaseModel):
    name: str = Field(
        ...,
        description="The name of the organization."
    )
    location: Optional[Location] = Field(
        ...,
        description="The location of the organization."
    )
    role: Optional[str] = Field(
        ...,
        description="The role of the organization in the contract."
    )

class Contract(BaseModel):
    contract_type: str = Field(
        ...,
        description="The type of contract being entered into.",
        enum=contract_types,        
    )
    parties: List[Organization] = Field(
        ...,
        description="List of parties involved in the contract, with details of each party's role."
    )
    effective_date: str = Field(
        ...,
        description="The date when the contract becomes effective. Use yyyy-MM-dd format.",
    )
    term: str = Field(
        ...,
        description="The duration of the agreement, including provisions for renewal or termination.",
    )
    contract_scope: str = Field(
        ...,
        description="Description of the scope of the contract, including rights, duties, and any limitations.",
    )
    end_date: Optional[str] = Field(
        ...,
        description="The date when the contract becomes expires. Use yyyy-MM-dd format.",
    )
    total_amount: Optional[float] = Field(
        ..., description="Total value of the contract."
    )
    governing_law: Optional[Location] = Field(
        ..., description="The jurisdiction's laws governing the contract."
    )