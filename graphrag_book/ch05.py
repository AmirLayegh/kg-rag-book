import os
import dotenv
import json
import ch05_tools
from utils import chat, tool_choice, neo4j_driver
from ch04 import create_movie_database
from dotenv import load_dotenv
load_dotenv()


query_update_prompt = """
    You are an expert at updating questions to make the them ask for one thing only, more atomic, specific and easier to find the answer for.
    You do this by filling in missing information in the question, with the extra information provided to you in previous answers. 
    
    You respond with the updated question that has all information in it.
    Only edit the question if needed. If the original question already is atomic, specific and easy to answer, you keep the original.
    Do not ask for more information than the original question. Only rephrase the question to make it more complete.
    
    JSON template to use:
    {
        "question": "question1"
    }
"""

def query_update(input: str, answers: list[any]) -> str: 
    messages = [
        {"role": "system", "content": query_update_prompt},
        *answers,
        {"role": "user", "content": f"The user question to rewrite: '{input}'"},
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages, model = "gpt-4o", config=config, )
    try:
        return json.loads(output)["question"]
    except json.JSONDecodeError:
        print("Error decoding JSON")
    return []


tool_picker_prompt = """Your job is to chose the right tool needed to respond to the user question.
The available tools are provided to you in the prompt.
Make sure to pass the right and the complete arguements to te chosen tool.
"""

tools = {
    "movie_info_by_title": {
        "description": ch05_tools.movie_info_by_title_description,
        "function": ch05_tools.movie_info_by_title,
    },
    "movies_info_by_actor": {
        "description": ch05_tools.movies_info_by_actor_description,
        "function": ch05_tools.movies_info_by_actor,
    },
    "text2cypher": {
        "description": ch05_tools.text2cypher_description,
        "function": ch05_tools.text2cypher,
    },
    "answer_given": {
        "description": ch05_tools.answer_given_description,
        "function": ch05_tools.answer_given,
    },
}

def handle_tool_calls(tools: dict[str, any], llm_tool_calls: list[dict[str, str]]):
    output = []
    if llm_tool_calls:
        for tool_call in llm_tool_calls:
            function_to_call = tools[tool_call.function.name]["function"]
            function_args = json.loads(tool_call.function.arguments)
            res = function_to_call(**function_args)
            output.append(res)
    return output

def route_question(question: str, tools: dict[str, any], answers: list[dict[str, any]]):
    llm_tool_calls = tool_choice(
        [
            {"role": "system", "content": tool_picker_prompt},
            *answers,
            {"role": "user", "content": f"The user question to route: '{question}'"},
        ],
        model="gpt-4o",
        tools=[tool["description"] for tool in tools.values()],
    )
    return handle_tool_calls(tools, llm_tool_calls)

def handle_user_input(input: str, answers: list[dict[str, str]]):
    updated_question = query_update(input, answers)
    response = route_question(updated_question, tools, answers)
    answers.append({"role": "assistant", "content": "For the question: '{updated_question}', we have the answer: '{json.dumps(response)}'"})
    return answers

answer_critique_prompt = """
    You are an expert at identifying if questions has been fully answered or if there is an opportunity to enrich the answer.
    The user will provide a question, and you will scan through the provided information to see if the question is answered.
    If anything is missing from the answer, you will provide a set of new questions that can be asked to gather the missing information.
    All new questions must be complete, atomic and specific.
    However, if the provided information is enough to answer the original question, you will respond with an empty list.

    JSON template to use for finding missing information:
    {
        "questions": ["question1", "question2"]
    }
"""

def critique_answers(question: str, answers: list[dict[str, str]]) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": answer_critique_prompt,
        },
        *answers,
        {
            "role": "user",
            "content": f"The original user question to answer: {question}",
        },
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages, model="gpt-4o", config=config)
    try:
        return json.loads(output)["questions"]
    except json.JSONDecodeError:
        print("Error decoding JSON")
    return []

main_prompt = """
    Your job is to help the user with their questions.
    You will receive user questions and information needed to answer the questions
    If the information is missing to answer part of or the whole question, you will say that the information 
    is missing. You will only use the information provided to you in the prompt to answer the questions.
    You are not allowed to make anything up or use external information.
"""

def agentic_rag(input: str):
    answers = []
    answers = handle_user_input(input, answers)
    critique = critique_answers(input, answers)

    if critique:
        answers = handle_user_input(" ".join(critique), answers)

    llm_response = chat(
        [
            {"role": "system", "content": main_prompt},
            *answers,
            {"role": "user", "content": f"The user question to answer: {input}"},
        ],
        model="gpt-4o",
    )

    return llm_response

if __name__ == "__main__":
    driver = neo4j_driver()
    #create_movie_database(driver)
    response = agentic_rag("Who's the main actor in the movie Matrix and what other movies is that person in?")
    print(response)
    driver.close()