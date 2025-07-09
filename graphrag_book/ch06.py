import dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
from openai import OpenAI
import json
from utils import neo4j_driver
import os

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
        description="The role of the organization in the contract, such as 'provider', 'client', 'supplier', etc."
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
    
system_prompt = """
You are an expert in extracting structured information from legal documents and contracts.
Identify key details such as parties involved, dates, terms, obligations, and legal definitions.
Present the extracted information in a clear, structured format. Beconcise, docusing on essential legal content and ignoring unnecesary boilerplate language. The extracted data will be used to address any questions that may arise reagarding the contracts.
"""

def extract(document, model="gpt-4o", temperature=0):
    response = client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": document},
        ],
        response_format=Contract,
    )
    return json.loads(response.choices[0].message.content)

def create_knowledge_graph(driver):
    driver.execute_query("""
                         CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contract) REQUIRE c.id IS UNIQUE;
                         """)
    driver.execute_query("""
                         CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE;
                         """)
    driver.execute_query("""
                         CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.fullAddress IS UNIQUE;
                         """)
    
def import_to_knowledge_graph(driver, document):
    import_query = """WITH $data AS contract_data
// Create Contract node
MERGE (contract:Contract {id: randomUUID()})
SET contract += {
  contract_type: contract_data.contract_type,
  effective_date: contract_data.effective_date,
  term: contract_data.term,
  contract_scope: contract_data.contract_scope,
  end_date: contract_data.end_date,
  total_amount: contract_data.total_amount,
  governing_law: contract_data.governing_law.state + ' ' +
                 contract_data.governing_law.country
}
WITH contract, contract_data
// Create Party nodes and their locations
UNWIND contract_data.parties AS party
MERGE (p:Organization {name: party.name})
MERGE (loc:Location {
  fullAddress: party.location.address + ' ' +
                party.location.city + ' ' +
                party.location.state + ' ' +
                party.location.country})
SET loc += {
  address: party.location.address,
  city: party.location.city,
  state: party.location.state,
  country: party.location.country
}
// Link party to their location
MERGE (p)-[:LOCATED_AT]->(loc)
// Link parties to the contract
MERGE (p)-[r:HAS_PARTY]->(contract)
SET r.role = party.role
"""
    driver.execute_query(import_query, data=extract(document))
    
def query_knowledge_graph(driver, query):
    cypher = """
    MATCH (c:Contract)
    WHERE c.contract_type = $query
    RETURN c
    """
    records, _, _ = driver.execute_query(cypher, query=query)
    return [record.data() for record in records]

if __name__ == "__main__":
    dotenv.load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open("data/license_agreement.txt", "r") as f:
        document = f.read()
    #print(extract(document))
    driver = neo4j_driver()
    create_knowledge_graph(driver)
    import_to_knowledge_graph(driver, document)
    print(query_knowledge_graph(driver, "Licensing Agreement"))
    
    
    
    