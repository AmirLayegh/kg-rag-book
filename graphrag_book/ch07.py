from utils import neo4j_driver, chat, chunk_text, embed, num_tokens_from_string
from dotenv import load_dotenv
import os
import requests
from ch07_tools import create_extraction_prompt, parse_extraction_output, import_nodes_query, import_relationships_query, get_summarize_prompt, import_entity_summary, import_rels_summary
from typing import List
from tqdm import tqdm
import neo4j

load_dotenv(override=True)

def load_data_and_chunk_into_books(file_path: str = "https://www.gutenberg.org/cache/epub/1727/pg1727.txt") -> list[str]:
    response = requests.get(file_path)
    text = ((response.text)
    .split("PREFACE TO FIRST EDITION")[2]
    .split("FOOTNOTES")[0]
    .strip()
    .split("\nBOOK")[1:])
    return text

def token_count(books: list[str]) -> int:
    token_count = [num_tokens_from_string(book) for book in books]
    print(
        f"""There are {len(token_count)} books in the corpus, with a total of {sum(token_count)} tokens and an average of {sum(token_count) / len(token_count)} tokens per book."""
    )

def chunk_books(books: list[str]) -> List[str]:
    chunk_size = 1000
    overlap = 40
    chunked_books = [chunk_text(book, chunk_size, overlap) for book in books]
    return chunked_books

ENTITY_TYPES = ["PERSON", 
                "ORGANIZATION", 
                "LOCATION", 
                "GOD", 
                "EVENT", 
                "CREATURE", 
                "WEAPON_OR_TOOL",
                "GEO"]

def extract_entities_and_relationships(text: str) -> List[str]:
    messages = [
        {"role": "user", "content": create_extraction_prompt(ENTITY_TYPES, text)}
    ]
    response = chat(messages, model="gpt-4o-mini")
    return parse_extraction_output(response)

def store_to_neo4j(driver, chunked_books: List[List[str]]):
    number_of_books = 1
    for book_i, book in enumerate(
        tqdm(chunked_books[:number_of_books], desc="Processing books")
    ):
        for chunk_i, chunk in enumerate(
            tqdm(book, desc="Processing chunks")
        ):
            entities, relationships = extract_entities_and_relationships(chunk)
            driver.execute_query(import_nodes_query, 
                                 data=entities,
                                 book_id=book_i,
                                 chunk_id=chunk_i,
                                 text=chunk)
            
            driver.execute_query(
                import_relationships_query,
                data=relationships,
                )
            
def query_database(driver: neo4j.Driver):
    data, _, _ =driver.execute_query("""
                         MATCH (:`__Entity__`)
                         RETURN 'entity' AS type, count(*) AS count
                         UNION
                         MATCH ()-[RELATIONSHIP]->()
                         RETURN 'relationship' AS type, count(*) AS count
                         """)
    print(data)
    
def query_person_description(driver: neo4j.Driver):
    data, _, _ = driver.execute_query("""
                                      MATCH (n:Person)
                                      WHERE n.name = 'TELEMACHUS'
                                      RETURN n.description as description
                                      """)
    print(data)

def query_relationship_description(driver: neo4j.Driver):
    data, _, _ = driver.execute_query("""
                                      MATCH(n:__Entity__)-[:RELATIONSHIP]-(m:__Entity__)
                                      WITH n, m, count(*) as countOfRels
                                      ORDER BY countOfRels DESC LIMIT 5
                                      MATCH (n)-[r:RELATIONSHIP]-(m)
                                      RETURN n.name as source, m.name as target, countOfRels, collect(r.description) as description
                                      """)
    print([el.data() for el in data])
    
def summarize_candidate_entities(driver: neo4j.Driver):
    candidate_to_summarize, _, _ = driver.execute_query("""
                                                        MATCH (e:__Entity__) WHERE size(e.description) > 1
                                                        RETURN e.name AS entity_name, e.description AS description_list
                                                        """)
    summaries = []
    for en in tqdm(candidate_to_summarize, desc="Summarizing entities"):
        messages = [
            #{"role": "system", "content": "You are a helpful assistant that summarizes the description of an entity."},
            {"role": "user", "content": get_summarize_prompt(en["entity_name"], en["description_list"])}
        ]
        response = chat(messages, model="gpt-4o-mini")
        summaries.append({"entity_name": en["entity_name"], "summary": response})
    return summaries

def import_summaries_to_neo4j(driver: neo4j.Driver, summaries: List[dict]):
    import_entity_summary(driver, summaries)
    
def query_summaries(driver: neo4j.Driver):
    data, _, _ = driver.execute_query("""
                                      MATCH (e:__Entity__)
                                      WHERE e.name = 'ATLAS'
                                      RETURN e.name AS entity_name, e.summary AS summary
                                      """)
    print([el.data() for el in data])   

def summarize_candidate_relationships(driver: neo4j.Driver):
    candidate_to_summarize, _, _ = driver.execute_query("""
                                                        MATCH (s:__Entity__)-[r:RELATIONSHIP]-(t:__Entity__)
                                                        WHERE id(s) < id(t)
                                                        WITH s.name AS source, t.name AS target, collect(r.description) AS description_list, count(*) as count
                                                        WHERE count > 1
                                                        RETURN source, target, description_list
                                                        """)
    summaries = []
    for rel in tqdm(candidate_to_summarize, desc="Summarizing relationships"):
        entity_name = f"{rel['source']} relationship to {rel['target']}"
        messages = [
            #{"role": "system", "content": "You are a helpful assistant that summarizes the description of a relationship."},
            {"role": "user", "content": get_summarize_prompt(entity_name, rel["description_list"])}
        ]
        response = chat(messages, model="gpt-4o-mini")
        summaries.append({"source": rel["source"], "target": rel["target"], "summary": response})
    return summaries

def import_relationship_summaries_to_neo4j(driver: neo4j.Driver, summaries: List[dict]):
    import_rels_summary(driver, summaries)
    
def query_relationship_summaries(driver: neo4j.Driver):
    data, _, _ = driver.execute_query("""  
                                      MATCH (s:__Entity__)-[r:SUMMARIZED_RELATIONSHIP]-(t:__Entity__)
                                      RETURN s.name AS source, t.name AS target, r.summary AS summary
                                      """)
    print([el.data() for el in data])

def query_relationship_summaries_by_source(driver: neo4j.Driver):
    pass
    

if __name__ == "__main__":
    books = load_data_and_chunk_into_books()
    token_count(books)
    chunked_books = chunk_books(books)
    #print(chunked_books[0][0])
    #embeddings = create_embeddings(chunks)
    driver = neo4j_driver()
    #driver.execute_query("""MATCH(n) DETACH DELETE(n)""")
    #store_to_neo4j(driver, chunked_books)
    #query_database(driver)
    #query_person_description(driver)
    #query_relationship_description(driver)
    #summaries = summarize_candidate_entities(driver)
    #import_summaries_to_neo4j(driver, summaries)
    #query_summaries(driver)
    #summaries = summarize_candidate_relationships(driver)
    #import_relationship_summaries_to_neo4j(driver, summaries)
    query_relationship_summaries(driver)
    driver.close()
            