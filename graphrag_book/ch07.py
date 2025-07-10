from utils import neo4j_driver, chat, chunk_text, embed, num_tokens_from_string
from dotenv import load_dotenv
import os
import requests
from ch07_tools import create_extraction_prompt, parse_extraction_output, import_nodes_query, import_relationships_query
from typing import List
from tqdm import tqdm

load_dotenv()

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

def extract_entities(text: str) -> List[str]:
    messages = [
        #{"role": "system", "content": "You are an expert in entity extraction from text."},
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
            entities, relationships = extract_entities(chunk)
            driver.execute_query(import_nodes_query, data=entities)
            driver.execute_query(import_relationships_query, data=relationships)
if __name__ == "__main__":
    books = load_data_and_chunk_into_books()
    token_count(books)
    chunked_books = chunk_books(books)
    #print(chunked_books[0][0])
    #embeddings = create_embeddings(chunks)
    
            