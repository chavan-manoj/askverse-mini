import logging
import json
from urllib.parse import urlparse
from colorama import Fore, Style
from askverse_mini.ai.llm import LLM
from askverse_mini.ai.planner_agent import PlannerAgent
from askverse_mini.api.rapid_apigateway import RapidAPIGateway

logger = logging.getLogger(__name__)

class AskVerse:
    """
    AskVerse is a smart AI system that identifies which tools/APIs are suitable, and execute only required ones.
    """
    def __init__(self, ask_docs, ask_wiki, ask_tavily, ask_arxiv):
        self.ask_system = {}
        self.ask_system["docs"] = ask_docs
        self.ask_system["wiki"] = ask_wiki
        self.ask_system["tavily"] = ask_tavily
        self.ask_system["arxiv"] = ask_arxiv

    def initialize(self):
        self.ask_system["docs"].initialize()
        self.ask_system["wiki"].initialize()
        self.ask_system["tavily"].initialize()
        self.ask_system["arxiv"].initialize()

        self.llm = LLM()
        self.planner_agent = PlannerAgent()
        self.gateway = RapidAPIGateway()
        
    def ask(self, question: str):
        plan = self.planner_agent.plan(question)
        if not plan or len(plan) == 0:
            return {
                "sources": [],
                "answer": "No relevant sources found for the given query/task."
            }
        self._print_plan(plan)
        
        confirmed = input("Proceed with these actions? You will be prompted for missing parameters. (type y|yes to continue): ").strip().lower()
        if confirmed.upper() not in ("Y", "YES"):
            return {
                "sources": [],
                "answer": "Discarded the plan."
            }
        param_values = {}
        for pl_action in plan:
            type = pl_action.get("type")
            if type == "api":
                action = pl_action["action"]
                if "missing_params" in action and action["missing_params"]:
                    for param in action["missing_params"]:
                        if param["name"] not in param_values:
                            value = input(f"Please provide value for '{param['name']}' ({param.get('description', '')}): ")
                            param_values[param["name"]] = value
        
        if len(param_values) > 0:
            for pl_action in plan:
                type = pl_action.get("type")
                if type == "api":
                    action = pl_action["action"]
                    url = action["url"]
                    for param_name, value in param_values.items():
                        url = url.replace(f"{{{param_name}}}", value)
                    action["url"] = url

            self._print_plan(plan)

        sources = []
        responses = []

        for pl_action in plan:
            type = pl_action.get("type")
            action = pl_action["action"]

            if type == "api":
                host = urlparse(action["server"]).netloc
                response = self.gateway.request(
                    host=host,
                    method=action["method"],
                    url=action["url"],
                    body=action.get("payload")
                )
                data = response.read()
                responses.append("\n```json\n" + str(data) + "\n```\n")
                sources.append(action["method"] + " " + action["server"] + action["url"])
                continue
            
            if type in ["docs", "wiki", "tavily", "arxiv"]:
                system = self.ask_system[type]
                response = system.ask(question)
                if "sources" in response:
                    sources.extend(response["sources"])
                if "answer" in response:
                    responses.append(response["answer"])
        
        types = [pl_action.get("type") for pl_action in plan]
        if "api" not in types and len(types) <= 1:
            return {
                "sources": sources,
                "answer": responses[0] if len(responses)>0 else "No answer found."
            }
        
        summarize_system_prompt = (
            "You expert at summarizing responses in concise and meaningful manner. "
            "Summarize the following search results in a concise and informative manner "
            "relevant to the exact user query provided below. "
            "Include additional details as appropriate to user query."
        )
        llm_input={
            "query": question,
            "search_results_json": responses
        }
        summarize_user_query = "Summarize the results. User query: {query} \n Raw results: {search_results_json} "
        summary = self.llm.query(system_prompt=summarize_system_prompt, user_query=summarize_user_query, input=llm_input)
        
        return {
            "sources": sources,
            "answer": summary
        }

    def _print_plan(self, plan):
        print(Fore.LIGHTMAGENTA_EX)
        print(f"Planned {len(plan)} actions:")
        for pl_action in plan:
            print(pl_action)
        print(Style.RESET_ALL)