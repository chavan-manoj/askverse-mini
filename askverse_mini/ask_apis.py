import logging
import json
from urllib.parse import urlparse
from askverse_mini.ai.llm import LLM
from askverse_mini.api.open_api_agent import OpenAPIAgent
from askverse_mini.api.rapid_apigateway import RapidAPIGateway

logger = logging.getLogger(__name__)

class AskAPIs:
    """
    A class to handle API calls for AskVerse Mini.
    """

    def __init__(self, oas_dir: str = "oas"):
        self.oas_dir = oas_dir

    def initialize(self):
        self.agent = OpenAPIAgent(oas_dir=self.oas_dir)
        self.gateway = RapidAPIGateway()
        self.llm = LLM()
        
    def ask(self, question: str):
        plan = self.agent.plan(question)
        print(f"Planned {len(plan)} API calls:")
        for action in plan:
            print(action)
        confirmed = input("Proceed with these actions? You will be prompted for missing parameters. (type yes to continue): ").strip().lower()
        if confirmed != "yes":
            return {
                "sources": [],
                "answer": "Discarded the plan."
            }
        param_values = {}
        for action in plan:
            if "missing_params" in action and action["missing_params"]:
                for param in action["missing_params"]:
                    if param["name"] not in param_values:
                        value = input(f"Please provide value for '{param['name']}' ({param.get('description', '')}): ")
                        param_values[param["name"]] = value
        
        if len(param_values) > 0:
            for action in plan:
                url = action["url"]
                for param_name, value in param_values.items():
                    url = url.replace(f"{{{param_name}}}", value)
                action["url"] = url

            print(f"Final Planned to execute {len(plan)} API calls:")
            for action in plan:
                print(action)

        sources = []
        responses = []

        for action in plan:
            host = urlparse(action["server"]).netloc
            response = self.gateway.request(
                host=host,
                method=action["method"],
                url=action["url"],
                body=action.get("payload")
            )
            data = response.read()
            responses.append(json.loads(data))
            sources.append(action["method"] + " " + action["server"] + action["url"])
        
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
        summarize_user_query = "Summarize the results. User query: {query} \n Movie search results: ```json\n{search_results_json}\n``` "
        summary = self.llm.query(system_prompt=summarize_system_prompt, user_query=summarize_user_query, input=llm_input)
        
        return {
            "sources": sources,
            "answer": summary
        }