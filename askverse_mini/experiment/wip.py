#WIP - Work In Progress
# This is a simple agentic AI assistant that can plan, collect inputs, confirm actions, and execute them.
# It uses a state graph to manage the flow of actions and a language model to generate responses.
# It also integrates with an external API to search for movies and provides a CLI for user interaction.
# The code is structured into several sections: tool stubs, agent state, planning, input collection, confirmation, execution, and the main CLI loop.
# The agent can perform actions like searching for flights, booking events, invoicing users, and searching for movie reviews.
# It uses the LangChain library for language model interactions and the LangGraph library for state management.
# This code is a work in progress and may not be fully functional or optimized. This will be integrated into the AskVerse framework.

import os
import json
import http.client
from typing import TypedDict, Optional, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import random
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME")
RAPID_API_KEY = os.getenv("RAPID_API_KEY")
IMDB_API_HOST = os.getenv("IMDB_API_HOST")

llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0
    )

def search_movie(query):
    conn = http.client.HTTPSConnection(IMDB_API_HOST)

    headers = {
        'x-rapidapi-key': RAPID_API_KEY,
        'x-rapidapi-host': IMDB_API_HOST
    }

    conn.request("GET", f"/imdb/autocomplete?query={query}", headers=headers)

    res = conn.getresponse()
    data = res.read()

    return json.loads(data)

def test_remote_api():
    query = input("Enter the movie name: ")
    results = search_movie(query)
    print("Here are the movies I found:")
    movies = [f"{movie['originalTitle']}" for movie in results]
    print(movies, sep="\n")

# === Tool Stubs ===
def search_flights(destination: str, date: str) -> str:
    return f"🛫 Found flights to {destination} on {date}. Booking link: http://book-flight/{random.randint(1000,9999)}"

def book_event(event_name: str, city: str, date: str) -> str:
    return f"🎟️ Booked event '{event_name}' in {city} on {date}. Link: http://event-ticket/{random.randint(1000,9999)}"

def invoice_user(name: str, email: str, amount: str) -> str:
    return f"💸 Invoice sent to {name} ({email}) for ${amount}. Invoice ID: INV-{random.randint(10000,99999)}"

def search_movie_reviews(movie_name: str) -> str:
    return f"🎬 Reviews for '{movie_name}': 9/10 IMDb, 88% Rotten Tomatoes"

TOOLS = {
    "search_flights": search_flights,
    "book_event": book_event,
    "invoice_user": invoice_user,
    "search_movie_reviews": search_movie_reviews
}

# === Agent State ===
class AgentState(TypedDict):
    user_input: str
    tool_plan: List[str]
    tool_inputs: dict
    confirmed: bool
    result: Optional[str]

# === Step 1: LLM Planning & Input Extraction ===
def plan_tools_and_inputs(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a smart assistant. Identify which tools to use for a user query and extract as many inputs as possible, "
         "specify None as input_value in tool_inputs if no input is already provided."
         ),
        ("user", 
         "User input: {input}\n\n"
         "Available tools:\n"
         "- search_flights(destination, date)\n"
         "- book_event(event_name, city, date)\n"
         "- invoice_user(name, email, amount)\n"
         "- search_movie_reviews(movie_name)\n\n"
         "Respond in JSON format:\n"
         "{{tool_plan: [...], tool_inputs: {{tool_name: {{input_name: input_value}}}}}}"
         )
    ])
    chain = prompt | llm
    response = chain.invoke({"input": state["user_input"]}).content
    parsed = eval(response)
    state["tool_plan"] = parsed["tool_plan"]
    state["tool_inputs"] = parsed["tool_inputs"]
    return state

# === Step 2: Collect Missing Inputs ===
def collect_tool_inputs(state: AgentState) -> AgentState:
    required_fields = {
        "search_flights": ["destination", "date"],
        "book_event": ["event_name", "city", "date"],
        "invoice_user": ["name", "email", "amount"],
        "search_movie_reviews": ["movie_name"]
    }
    for tool in state["tool_plan"]:
        for field in required_fields.get(tool, []):
            tool_input = state["tool_inputs"].get(tool, {})
            if field not in tool_input or tool_input[field] is None:
                value = input(f"[{tool}] Please provide {field}: ")
                state["tool_inputs"].setdefault(tool, {})[field] = value
    return state

# === Step 3: Confirm Plan ===
def confirm_plan(state: AgentState) -> AgentState:
    print("\n🔍 Planned Actions:")
    for tool in state["tool_plan"]:
        inputs = state["tool_inputs"].get(tool, {})
        print(f"→ {tool} with inputs: {inputs}")
    confirmed = input("\n✅ Proceed with these actions? (yes/no): ").strip().lower()
    state["confirmed"] = confirmed == "yes"
    return state

# === Step 4: Execute Plan ===
def execute_plan(state: AgentState) -> AgentState:
    if not state["confirmed"]:
        state["result"] = "❌ Cancelled by user."
        return state
    results = []
    for tool in state["tool_plan"]:
        fn = TOOLS[tool]
        inputs = state["tool_inputs"][tool]
        results.append(fn(**inputs))
    state["result"] = "\n\n".join(results)
    return state

# === LangGraph ===
graph = StateGraph(AgentState)
graph.set_entry_point("Plan")
graph.add_node("Plan", plan_tools_and_inputs)
graph.add_node("CollectInputs", collect_tool_inputs)
graph.add_node("Confirm", confirm_plan)
graph.add_node("Execute", execute_plan)

graph.add_edge("Plan", "CollectInputs")
graph.add_edge("CollectInputs", "Confirm")
graph.add_edge("Confirm", "Execute")
graph.add_edge("Execute", END)

app = graph.compile()

# === Main CLI Loop ===
def main():
    print("🤖 Welcome to the Agentic AI Assistant!")
    while True:
        user_input = input("\n📝 What can I help you with?\n> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        state = {
            "user_input": user_input,
            "tool_plan": [],
            "tool_inputs": {},
            "confirmed": False,
            "result": None
        }
        result = app.invoke(state)
        print("\n🎯 Result:\n" + result["result"])


if __name__ == "__main__":
    main()
