import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import tiktoken

load_dotenv(override=True)

open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_ne4j_index(driver, index_name, embeddings):
    driver.execute_query(
        f"""CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {len(embeddings[0])},
                `vector.similarity_function`: 'cosine'
            }}
        }}"""
    )

def neo4j_driver():
    return GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

def clear_existing_data(driver):
    """Clear existing PDF data to avoid embedding dimension conflicts"""
    clear_query = """
    MATCH (pdf:PDF {id: '1709.00666'})
    DETACH DELETE pdf
    """
    try:
        driver.execute_query(clear_query)
        print("Cleared existing data")
    except Exception as e:
        print(f"Error clearing data: {e}")

def drop_vector_index(driver, index_name):
    """Drop existing vector index if it exists"""
    try:
        driver.execute_query(f"DROP INDEX {index_name} IF EXISTS")
        print(f"Dropped existing index: {index_name}")
    except Exception as e:
        print(f"Error dropping index: {e}")
        
def chunk_text(text: str, chunk_size: int, overlap: int, split_on_whitespaces: bool = True) -> list[str]:
    """
    Chunk text into chunks of a given size, with an overlap.
    """
    chunks = []
    index = 0
    while index < len(text):
        if split_on_whitespaces:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index = end
    return chunks

def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def embed(text, model):
    if model == "text-embedding-3-small":  
        response = open_ai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return list(map(lambda x: x.embedding, response.data))
    elif model == "all-MiniLM-L12-v2":
        model = SentenceTransformer("all-MiniLM-L12-v2")
        return list(map(lambda x: model.encode(x), text))

def chat(messages, model="gpt-4o-mini", temp=0.0, config={}):
    response = open_ai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        **config
    )
    return response.choices[0].message.content

def tool_choice(messages, model="gpt-4o", temperature=0, tools=[], config={}):
    response = open_ai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        tools=tools or None,
        **config,
    )
    return response.choices[0].message.tool_calls