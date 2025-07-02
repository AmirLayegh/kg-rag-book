import requests
import pdfplumber
from utils import chunk_text
from openai import OpenAI
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
load_dotenv(override=True)

open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

remote_pdf_url = "https://arxiv.org/pdf/1709.00666.pdf"
prf_filename = "ch02-downloaded.pdf"

def download_and_create_pdf_chunks(url: str, pdf_name: str, chunk_size: int, overlap: int, split_on_whitespaces: bool = True) -> list[str]:
    response = requests.get(url)
    if response.status_code == 200:
        with open(pdf_name, "wb") as f:
            f.write(response.content)
        with pdfplumber.open(pdf_name) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return chunk_text(text, chunk_size, overlap, split_on_whitespaces)
    else:
        print(f"Failed to download PDF from {url}")
        return []
    
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

from neo4j import GraphDatabase

def create_neo4j_index(driver, index_name, embeddings):
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

def store_chunks_and_populate_index(driver, chunks, embeddings):
    driver.execute_query(
        """
        WITH $chunks as chunks, range(0, size($chunks)) AS index
        UNWIND index AS i
        WITH i, chunks[i] AS chunk, $embeddings[i] AS embedding
        MERGE (c:Chunk {index: i})
        SET c.text = chunk, c.embedding = embedding
        """,
        chunks=chunks, 
        embeddings=embeddings
    )
    
def get_data_form_chunk(driver, chunk_index):
    results, _, _ = driver.execute_query(
        f"""
        MATCH (c:Chunk)
        WHERE c.index = {chunk_index}
        RETURN c.text AS text, c.embedding AS embedding
        """
    )
    return results[0]["text"], results[0]["embedding"]

def embed_question(question, model):
    question_embedding = embed([question], model)
    return question_embedding[0]

def vector_similarity_search(driver, index_name, k = 5, question_embedding = None):
    similar_records, _, _ = driver.execute_query(
        f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $question_embedding) YIELD node AS hits, score
        RETURN hits.text AS text, score, hits.index AS index
        """,
        k=k,
        question_embedding=question_embedding
    )
    return similar_records

def generate_answer(similar_records, question):
    system_message = """You are an Einstein expert, but can only use the provided documents to respons the questions."""
    
    # Extract text from records, handling both data structures
    documents = []
    for doc in similar_records:
        try:
            if hasattr(doc, 'keys') and 'text' in doc.keys():
                # Vector search format: {"text": "...", "score": ..., "index": ...}
                documents.append(doc["text"])
            elif hasattr(doc, 'keys') and 'node' in doc.keys():
                # Hybrid search format: Neo4j Record with node
                node = doc["node"]
                if hasattr(node, 'get'):
                    # Neo4j Node object with .get() method
                    text = node.get("text", "")
                    documents.append(text)
                elif hasattr(node, '__getitem__'):
                    # Dictionary-like access
                    text = node["text"]
                    documents.append(text)
                else:
                    # Try direct property access
                    text = getattr(node, 'text', '')
                    documents.append(text)
            else:
                # Try direct access for regular dictionaries
                if "text" in doc:
                    documents.append(doc["text"])
                elif "node" in doc:
                    node = doc["node"]
                    if "text" in node:
                        documents.append(node["text"])
        except Exception as e:
            print(f"Error extracting text from record: {e}")
            continue
    
    user_message = f"""Use the following documents to answer the question that will follow:
    {documents}
    
    ---
    The question to answer using information only from the above documents:
    {question}
    """
    
    print(f"Question: {question}")
    # print("--------------------------------\n")
    # print(f"user_message: {user_message}")
    # print("--------------------------------\n")
    stream = open_ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    #return chunk.choices[0].message.content
    
def create_full_text_index(driver):
    driver.execute_query(
        """
        CREATE FULLTEXT INDEX pdfChunkFulltext IF NOT EXISTS FOR (c:Chunk)
        ON EACH [c.text]
        """
    )

def hybrid_search(driver, index_name, full_text_index_name, question, k = 5, question_embedding = None):
    similar_hybrid_records, _, _ = driver.execute_query(
        f"""
        CALL {{
            CALL db.index.vector.queryNodes('{index_name}', $k, $question_embedding) YIELD node, score
            WITH collect({{node:node, score:score}}) AS nodes, max(score) AS max
            UNWIND nodes AS n
            //normalize scores
            RETURN n.node AS node, (n.score / max) AS score
            UNION
            //keyword index
            CALL db.index.fulltext.queryNodes('{full_text_index_name}', $question, {{limit: $k}}) YIELD node, score
            WITH collect({{node:node, score:score}}) AS nodes, max(score) AS max
            UNWIND nodes AS n
            RETURN n.node AS node, (n.score / max) AS score
        }}
        //deduplicate nodes
        WITH node, max(score) AS score ORDER BY score DESC LIMIT $k
        RETURN node, score
        """,
        k=k,
        question_embedding=question_embedding,
        question=question
    )
    return similar_hybrid_records
    
if __name__ == "__main__":
    chunks = download_and_create_pdf_chunks(remote_pdf_url, prf_filename, 500, 40, True)
    embeddings = embed(chunks, "all-MiniLM-L12-v2")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Number of embedding dimensions: {len(embeddings[0])}")
    #print(f"First embedding: {embeddings[0]}")
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    create_neo4j_index(driver, "pdf", embeddings)
    store_chunks_and_populate_index(driver, chunks, embeddings)
    text, embedding = get_data_form_chunk(driver, 0)
    #print(f"Text: {text}")
    #print(f"Embedding: {embedding}")
    question = "At what time was Einstein really interested in experimental works?"
    question_embedding = embed_question(question, "all-MiniLM-L12-v2")
    similar_records = vector_similarity_search(driver, "pdf", 5, question_embedding)
    # for record in similar_records:
    #     print(f"Text: {record['text']}")
    #     print(f"Score: {record['score']}")
    #     print(f"Index: {record['index']}")
    #     print("--------------------------------")
    #answer = generate_answer(similar_records, question)
    create_full_text_index(driver)
    similar_hybrid_records = hybrid_search(driver, "pdf", "pdfChunkFulltext", question, 5, question_embedding)
    #print(f"similar_hybrid_records: {similar_hybrid_records}")
    answer = generate_answer(similar_hybrid_records, question)
    
    driver.close()
