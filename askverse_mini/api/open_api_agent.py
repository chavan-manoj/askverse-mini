import os
import yaml
import glob
from typing import List, Dict, Any
from askverse_mini.ai.llm import LLM

class OpenAPIAgent:
    """
    An agent that loads all OpenAPI spec YAML files from the oas directory,
    and, given a user question or task, determines which APIs/methods to call,
    in what sequence, with which parameters, and what further inputs are required.
    """
    def __init__(self, oas_dir: str = "oas"):
        self.oas_dir = oas_dir
        self.llm = LLM()
        self.api_specs = self._load_all_specs()

    def _load_all_specs(self) -> List[Dict[str, Any]]:
        specs = []
        for file in glob.glob(os.path.join(self.oas_dir, "*.yaml")) + glob.glob(os.path.join(self.oas_dir, "*.yml")):
            with open(file, "r", encoding="utf-8") as f:
                specs.append(yaml.safe_load(f))
        return specs

    def _oas_summary(self) -> str:
        """
        Returns a summary string of all loaded OpenAPI specs for LLM context.
        """
        summaries = []
        for spec in self.api_specs:
            title = spec.get("info", {}).get("title", "Unknown API")
            servers = [s["url"] for s in spec.get("servers", [])] if "servers" in spec else []
            for path, methods in spec.get("paths", {}).items():
                for method, details in methods.items():
                    params = details.get("parameters", [])
                    param_str = ", ".join([f"{p['name']} ({p['schema']['type']})" for p in params if "name" in p and "schema" in p])
                    summaries.append(
                        f"API: {title}\nServer(s): {servers}\nPath: {path}\nMethod: {method.upper()}\nParameters: {param_str}\n"
                    )
        return "\n".join(summaries)

    def plan(self, user_query: str) -> Dict[str, Any]:
        """
        Given a user query, returns a plan of API calls to accomplish the task.
        """
        system_prompt = (
            "You are an expert API planner. You have access to the following OpenAPI specs:\n"
            f"{self._oas_summary()}\n"
            "Given a user question or task, identify exactly which APIs and methods need to be executed, "
            "in which sequence, and extract the parameters already specified by the user. "
            "List any mandatory parameters that are still required to execute the API(s). "
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
            "Ensure the JSON is valid and does not contain any code block markers.\n"
            "\n\nExample query 'How is the weather in Bengaluru' should respond with:\n"
            "[\n"
            "  {{\n"
            "    \"server\": \"https://weatherapi-com.p.rapidapi.com\",\n"
            "    \"method\": \"GET\",\n"
            "    \"url\": \"/forecast.json?q=Bengaluru&days={{days}}\",\n"
            "    \"payload\": null,\n"
            "    \"missing_params\": [{{\"name\":\"days\", \"description\": \"Number of days for the forecast (1-10)\"}}]\n"
            "  }}\n"
            "]"
            "\n\nExample output for query 'Name top 5 hindi comedy movies rated above 7 created before 1980 and has more than 1K votes' should respond with:\n"
            "[\n"
            "  {{\n"
            "    \"server\": \"https://imdb236.p.rapidapi.com\",\n"
            "    \"method\": \"GET\",\n"
            "    \"url\": \"/api/imdb/search?type=movie&genre=Comedy&averageRatingFrom=7&numVotesFrom=1000&startYearTo=1979&rows=5&spokenLanguages=hi\",\n"
            "    \"payload\": null,\n"
            "    \"missing_params\": [{{\"name\":\"days\", \"description\": \"Number of days for the forecast (1-10)\"}}]\n"
            "  }}\n"
            "]\n\n"
        )
        user_prompt = f"User query: {user_query}\nRespond ONLY with the JSON plan as described above."
        response = self.llm.query(system_prompt, user_prompt)
        # Remove code block markers if present
        import re, json
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
