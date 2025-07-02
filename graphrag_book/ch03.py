import dotenv
import re
from typing import List

import pdfplumber
import requests
from openai import OpenAI
import os
import tiktoken
from utils import chunk_text, chat, neo4j_driver, embed, clear_existing_data, drop_vector_index

from dotenv import load_dotenv

load_dotenv(override=True)

remote_pdf_url = "https://arxiv.org/pdf/1709.00666.pdf"
prf_filename = "ch03-downloaded.pdf"

open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

stepback_sysyem_promtp = """
You are an expert at world knowledge. Your task is to step back
and paraphrase a question to a more generic step-back question, which
is easier to answer. Here are a few examples

"input": "Could the members of The Police perform lawful arrests?"
"output": "what can the members of The Police do?"

"input": "Jan Sindel's was born in what country?"
"output": "what is Jan Sindel's personal history?"
"""

def generate_stepback_question(question):
    user_message = f"""{question}"""
    stepback_question = chat(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": stepback_sysyem_promtp},
            {"role": "user", "content": user_message}
        ],
    )
    return stepback_question

def download_and_create_pdf(url: str, pdf_name: str) -> list[str]:
    response = requests.get(url)
    if response.status_code == 200:
        with open(pdf_name, "wb") as f:
            f.write(response.content)
        with pdfplumber.open(pdf_name) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    else:
        print(f"Failed to download PDF from {url}")
        return []
    
# parent document retrieval
def split_text_by_title(text: str) -> list[str]:
    title_pattern = re.compile(r"(\n\d+[A-Z]?\. {1,3}.{0,60}\n)", re.DOTALL)
    titles = title_pattern.findall(text)
    # split the text into sections based on titles
    sections = re.split(title_pattern, text)
    sections_with_titles = []
    sections_with_titles.append(sections[0])
    for i in range(1, len(titles)+1):
        section_text = sections[i * 2 - 1].strip() + "\n" + sections[i * 2].strip()
        sections_with_titles.append(section_text)
    return sections_with_titles

def num_tokens_from_section(section: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(section)
    return len(tokens)

def create_parent_chunks(sections: list[str]) -> list[str]:
    parent_chunks = []
    for section in sections:
        parent_chunks.extend(chunk_text(section, 2000, 40))
        # if num_tokens_from_section(section) > max_tokens:
        #     chunks = chunk_text(section, max_tokens, 40)
        #     parent_chunks.extend(chunks)  # extend instead of append to flatten the list
        # else:
        #     parent_chunks.append(section)
    return parent_chunks

def store_parent_chunks(driver, parent_chunks):
    cypher_import_query = """
    MERGE (pdf:PDF {id:$pdf_id})
    MERGE (p:Parent {id:$pdf_id + '-' + $id})
    SET p.text = $parent
    MERGE (pdf)-[:HAS_PARENT]->(p)
    WITH p, $children AS children, $embeddings as embeddings
    UNWIND range(0, size(children) - 1) AS child_index
    MERGE (c:Child {id: $pdf_id + '-' + $id + '-' + toString(child_index)})
    SET c.text = children[child_index], c.embedding = embeddings[child_index]
    MERGE (p)-[:HAS_CHILD]->(c);
"""
    #driver = neo4j_driver()
    for i, chunk in enumerate(parent_chunks):
        child_chunk = chunk_text(chunk, 500, 20)
        embeddings = embed(child_chunk, "all-MiniLM-L12-v2")
        driver.execute_query(cypher_import_query, 
                             pdf_id="1709.00666", 
                             parent=chunk,
                             id = str(i),
                             children=child_chunk, 
                             embeddings=embeddings)
    #driver.close()
def create_vector_index_on_child_nodes(driver):
    index_name = "parent"
    try:
        driver.execute_query(f"""CALL db.index.vector.createNodeIndex($index_name, 'Child', 'embedding', 384, 'cosine')""", index_name=index_name)
    except Exception as e:
        print(f"Error creating vector index on child nodes: {e}")

def parent_retrieval(driver, question, index_name):
    question_embedding = embed([question], "all-MiniLM-L12-v2")[0]
    retrieval_query = """CALL db.index.vector.queryNodes($index_name, $k * 4, $question_embedding)
                            YIELD node, score
                            MATCH (node)<-[:HAS_CHILD]-(parent)
                            WITH parent, max(score) AS score
                            RETURN parent.text AS text, score
                            ORDER BY score DESC
                            LIMIT toInteger($k)
                            """
    similar_records, _, _ = driver.execute_query(retrieval_query, index_name=index_name, question_embedding=question_embedding, k=10)
    return [record["text"] for record in similar_records]

def generate_answer(question: str, documents: List[str]) -> str:
    answer_system_message = "You're en Einstein expert, but can only use the provided documents to respond to the questions."

    user_message = f"""
    Use the following documents to answer the question that will follow:
    {documents}

    ---

    The question to answer using information only from the above documents: {question}
    """
    result = chat(
        messages=[
            {"role": "system", "content": answer_system_message},
            {"role": "user", "content": user_message},
        ]
    )
    #print("Response:", result.choices[0].message.content)
    return result

def rag_pipeline(question: str):
    stepback_question = generate_stepback_question(question)
    print(f"Stepback question: {stepback_question}")
    
    driver = neo4j_driver()
    
    # Clear existing data to avoid embedding dimension conflicts
    clear_existing_data(driver)
    drop_vector_index(driver, "parent")
    
    text = download_and_create_pdf(remote_pdf_url, prf_filename)
    sections = split_text_by_title(text)
    print(f"Number of sections: {len(sections)}")
    parent_chunks = create_parent_chunks(sections)
    print(f"Number of parent chunks: {len(parent_chunks)}")
    store_parent_chunks(driver, parent_chunks)
    create_vector_index_on_child_nodes(driver)
    similar_documents = parent_retrieval(driver, stepback_question, "parent")
    answer = generate_answer(question, similar_documents)
    print(answer)
    
    driver.close()


if __name__ == "__main__":
    question = "When was Einstein granted the patent for his blouse design?"
    stepback_question = generate_stepback_question(question)
    print(f"Stepback question: {stepback_question}")
    
    driver = neo4j_driver()
    
    # Clear existing data to avoid embedding dimension conflicts
    #clear_existing_data(driver)
    #drop_vector_index(driver, "parent")
    
    text = download_and_create_pdf(remote_pdf_url, prf_filename)
    sections = split_text_by_title(text)
    #print(f"Number of sections: {len(sections)}")
    # for i, section in enumerate(sections):
    #     print(f"Section {i}: {num_tokens_from_section(section)} tokens")
    parent_chunks = create_parent_chunks(sections)
    #print(f"Number of parent chunks: {len(parent_chunks)}")
    # for i, chunk in enumerate(parent_chunks):
    #     print(f"Parent chunk {i}: {num_tokens_from_section(chunk)} tokens")
    
    #store_parent_chunks(driver, parent_chunks)
    #create_vector_index_on_child_nodes(driver)
    similar_documents = parent_retrieval(driver, stepback_question, "parent")
    for i, doc in enumerate(similar_documents):
        print(f"Similar document {i}: {doc}")
        print("--------------------------------")
    #print(similar_records)
    answer = generate_answer(question, similar_documents)
    print(answer)
    
    # Close the driver properly at the end
    driver.close()

