from cypher_queries import movie_query
from utils import neo4j_driver
from schema_utils import get_schema, chat

from dotenv import load_dotenv
load_dotenv()


def create_movie_database(driver):
    statements = [stmt.strip() for stmt in movie_query.split(';') if stmt.strip()]

    for i, statement in enumerate(statements):
        print(f"Executing statement {i+1}/{len(statements)}")
        try:
            driver.execute_query(statement)
        except Exception as e:
            print(f"Error executing statement {i+1}: {e}")
            print(f"Statement was: {statement[:100]}...")
            break

    print("Database setup completed successfully!")

def print_schema(driver):
    schema = get_schema(driver)
    print(schema)

def create_full_prompt(driver, question):
        prompt_template = """
        Instructions: 
        Generate Cypher statement to query a graph database to get the data to answer the user question below.

        Graph Database Schema:
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided in the schema.
        {schema}

        Terminology mapping:
        This section is helpful to map terminology between the user question and the graph database schema.
        {terminology}

        Examples:
        The following examples provide useful patterns for querying the graph database.
        {examples}

        Format instructions:
        Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to 
        construct a Cypher statement.
        Do not include any text except the generated Cypher statement.
        ONLY RESPOND WITH CYPHER, NO CODEBLOCKS.

        User question: {question}
        """
        schema_string = get_schema(driver)
        terminology_string = """
        Persons: When a user asks about a person by trade like actor, writer, director,
        producer, reviewer, they are referring to a node with the label 'Person'.
        Movies: When a user asks about a film or movie, they are referring to a node with the
        label Movie.
        """
        examples = [["Who are the two people acted in most movies together?", "MATCH\
            (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person) WHERE p1 <> p2 RETURN\
                p1.name, p2.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1"]]
        
        full_prompt = prompt_template.format(question=question, schema=schema_string,
                                            terminology=terminology_string,examples="\n".join([f"Question: {e[0]}\nCypher: {e[1]}"
                                                                                                for i, e in enumerate(examples)]))
        return full_prompt

if __name__ == "__main__":
    driver = neo4j_driver()
    #create_movie_database(driver)
    #print_schema(driver)
    full_prompt = create_full_prompt(driver, "Who directed the most movies?")
    #print(full_prompt)
    response = chat(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates Cypher statements to query a graph database."},
            {"role": "user", "content": full_prompt}
        ],
    )
    print(response)
    driver.close()
