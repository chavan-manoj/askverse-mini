import os
import yaml
import json
import re
import glob
from typing import List, Dict, Any
from askverse_mini.ai.llm import LLM

class PlannerAgent:
    """
    An agent that loads all OpenAPI spec YAML files from the oas directory,
    loads metadata about the tools, when to invoke them,
    and, given a user question or task, determines which Tools or APIs/methods to call,
    in what sequence, with which parameters, and what further inputs are required.
    """
    def __init__(self, oas_dir: str = "oas", tools_metadata_path = "metadata/tools.json", samples_path: str = "metadata/samples.json"):
        self.oas_dir = oas_dir
        self.tools_metadata_path = tools_metadata_path
        self.samples_path = samples_path

        self.llm = LLM()
        self.api_specs = "".join(self._load_all_specs_as_strings_for_llm())

        self.tools_metadata = self._load_json_as_string_for_llm(self.tools_metadata_path)
        self.samples = self._load_json_as_string_for_llm(self.samples_path)

    def _load_json_as_string_for_llm(self, filepath) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
            json_str = json.dumps(json_dict, indent=4).replace("{", "{{").replace("}", "}}")
            return "\n\n ```json\n" + json_str + "\n```\n"

    def _load_yaml_as_string_for_llm(self, filepath) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f)
            yaml_str = yaml.dump(yaml_dict).replace("{", "{{").replace("}", "}}")
            return "\n\n ```yaml\n" + yaml_str + "\n```\n"

    def _load_all_specs_as_strings_for_llm(self) -> List[Dict[str, Any]]:
        specs = []
        for file in glob.glob(os.path.join(self.oas_dir, "*.yaml")) + glob.glob(os.path.join(self.oas_dir, "*.yml")):
            specs.append(self._load_yaml_as_string_for_llm(file))
        return specs

    def plan(self, user_query: str) -> Dict[str, Any]:
        """
        Given a user query, returns a plan of API calls to accomplish the task.
        """
        system_prompt = (
            "You are an expert AI orchestration agent. You are also an API expert who very well understands Open API Specification. "
            "Carefully understand each tool and API definitions (given below). "
            "Your job is to carefully analyze a user’s question or task, identify which tools and APIs are relevant to the user query (or part of the user query). "
            "If required, break down user query into a minimal set of subqueries, "
            "and plan the exact sequence of tools and API calls required to answer the user query or accomplish the user task. "
            "When you break down the user query into subqueries, ensure the pronouns and relative terms in the sub query are resolved, "
            "You have access to the following tools, look at the description of each tool to understand when they are relevant, "
            "and refer examples given in same yaml to better understand the scenarios when those tools are relevant. "
            "Decide whether to use docs tool or not based on the metadata of documents it can search on. "
            "The tools.yaml is given below:\n" + self.tools_metadata + "\n\n"
            "In addition to these tools, you also have access to APIs which can help answer the user query or accomplish user task. "
            "Given below are the Open API specs of APIs you have access to.\n"
            "Understand each API thoroughly. Understand when to use these APIs based on endpoints, its description, summary, parameters, and other details. "
            "Also understand what each API endpoint responds with, that will help you understand dependency between APIs and Tools, "
            "and how to use the response of one API as input to another API/Tool. "
            "The Open API specs are given below as a set of yaml files:\n" + self.api_specs + "\n\n"
            "Given a user question or task, identify exactly which Tools or APIs need to be executed, in which sequence. "

            "In case you identify suitable APIs that can do the job then:\n"
            "------------------------------------------------\n"
            "automatically extract the API parameters already specified by the user. "
            "List any mandatory parameters that are still required to execute the API(s). "
            "Return an empty list if the user query is unrelated to these API definitions or no API can meet the tasks/subtasks given as user query. "
            "Automatically convert the parameters to appropriate data types, formats and enum values (wherever applicable), "
            "Identify if the parameter values are single values or lists, based on that identify the most appropriate parameter. "
            "e.g., use genre parameter for singular value and genres for plural parameter (if defined in schema). "
            "If the value is singular and no singular parameter is available, use the plural parameter and format data as plural based on the schema. "
            "Automatically detect language, country, city, date, time, in various formats from the user input. "
            "If the parameter description says specifically about the format of data "
            "(e.g., date, city, country code, language code etc), automatically convert the user input to that format. "
            "e.g., Hindi ISO language code is 'hi', India country code is 'IN', Bangalore city's new name is 'Bengaluru'. "
            "Understand words like today, tomorrow, etc and translate them as appropriate. "
            "Return a JSON array of actions, each with: server, method, url "
            "(with parameters filled in if available, otherwise use {{param}}), and payload (null for GET). "
            "Also, for each action, list any required parameters that are missing from the user query, "
            "so the caller can prompt the user for them before execution. "
            "------------------------------------------------\n"
            "Ensure the JSON is valid and does not contain any code block markers.\n"
            "Please refer example_query and example_response attributes in the following yaml snippet to understand what to respond in case of various user queries\n\n"
            "Even for APIs, you need to return a JSON plan with structure defined under example_response attribute:\n"
            "\n\n" + self.samples + "\n"
        )
        user_prompt = f"User query: {user_query}\nRespond ONLY with the JSON plan as described above."
        response = self.llm.query(system_prompt, user_prompt)
        # Remove code block markers if present
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"^```[a-zA-Z]*\n?", "", response)
            response = response.rstrip("`").strip()
        try:
            plan = json.loads(response)
        except Exception as e:
            print("Could not parse LLM response as JSON:", e)
            print(response)
            plan = []
        return plan
