from utils import neo4j_driver
from text2cypher import Text2Cypher

text2cypher_description = {
    "type": "function",
    "function": {
        "name": "text2cypher",
        "description": "Query the database with a user question. When other tools don't fit, fallback to use this one.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", 
                             "description": "The user question to find the answer for"
                             },
            },
        "required": ["question"]    
        },
    },
}

def text2cypher(question: str):
    driver = neo4j_driver()
    t2c = Text2Cypher(driver)
    t2c.set_prompt_section("question", question)
    cypher = t2c.generate_cypher()
    records, _, _ = driver.execute_query(cypher)
    result = [record.data() for record in records]
    driver.close()
    return result

movie_info_by_title_description = {
    "type": "function",
    "function": {
        "name": "movie_info_by_title",
        "description": "Get information about a movie by its title",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", 
                          "description": "The title of the movie"
                          },
            },
        "required": ["title"],
        },
    },
}

def movie_info_by_title(title: str):
    cypher = """
 MATCH (m:Movie)
 WHERE toLower(m.title) CONTAINS $title
 OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
 OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
 RETURN m AS movie, collect(a.name) AS cast, collect(d.name) AS directors
 """
    driver = neo4j_driver()
    records, _, _ = driver.execute_query(cypher, title=title.lower())
    result = [record.data() for record in records]
    driver.close()
    return result

movies_info_by_actor_description = {
    "type": "function",
    "function": {
        "name": "movies_info_by_actor",
        "description": "Get information about a movie by its actor",
        "parameters": {
            "type": "object",
            "properties": {
                "actor": {"type": "string", 
                          "description": "The name of the actor"
                          },
            },
        "required": ["actor"],
        },
    },
}

def movies_info_by_actor(actor: str):
    cypher = """
    MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
    OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(p:Person)
    WHERE toLower(a.name) CONTAINS $actor
    RETURN m AS movie, collect(a.name) AS cast, collect(d.name) AS directors
 """
    driver = neo4j_driver()
    records, _, _ = driver.execute_query(cypher, actor=actor.lower())
    result = [record.data() for record in records]
    driver.close()
    return result

answer_given_description = {
    "type": "function",
    "function": {
        "name": "answer_given",
        "description": "If a complete answer to the question already is provided in the conversation, use this tool to extract it.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the question",
                },
            },
            "required": ["answer"],
        },
    },
}

def answer_given(answer: str):
    return answer