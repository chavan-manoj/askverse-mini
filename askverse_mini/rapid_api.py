import os
import json
import urllib.parse
from askverse_mini.llm import LLM
from askverse_mini.rapid_apigateway import RapidAPIGateway

class RapidAPI:
    def __init__(self):
        self.IMDB_API_HOST = os.getenv("IMDB_API_HOST")
        self.WEATHER_API_HOST = os.getenv("WEATHER_API_HOST")
        self.gateway = RapidAPIGateway()

    def search_movie_name(self, query):
        query = urllib.parse.quote(query, safe='')
        url = f"/api/imdb/autocomplete?query={query}"
        res = self.gateway.request(
            host=self.IMDB_API_HOST,
            method="GET",
            url=url
        )
        data = res.read()
        return json.loads(data)
    
    def get_weather(self, city, days=3):
        city = urllib.parse.quote(city, safe='')
        url = f"/forecast.json?q={city}&days={days}"
        res = self.gateway.request(
            host=self.WEATHER_API_HOST,
            method="GET",
            url=url
        )
        data = res.read()
        return json.loads(data)

    def search_movie_advanced(
        self,
        originalTitle=None,
        originalTitleAutocomplete=None,
        primaryTitle=None,
        primaryTitleAutocomplete=None,
        type=None,
        genre=None,
        genres=None,
        isAdult=None,
        averageRatingFrom=None,
        averageRatingTo=None,
        numVotesFrom=None,
        numVotesTo=None,
        rows=None,
        startYearFrom=None,
        startYearTo=None,
        countriesOfOrigin=None,
        spokenLanguages=None,
        sortOrder=None,
        sortField=None
    ):
        params = {
            "originalTitle": originalTitle,
            "originalTitleAutocomplete": originalTitleAutocomplete,
            "primaryTitle": primaryTitle,
            "primaryTitleAutocomplete": primaryTitleAutocomplete,
            "type": type,
            "genre": genre,
            "genres": genres,
            "isAdult": isAdult,
            "averageRatingFrom": averageRatingFrom,
            "averageRatingTo": averageRatingTo,
            "numVotesFrom": numVotesFrom,
            "numVotesTo": numVotesTo,
            "rows": rows,
            "startYearFrom": startYearFrom,
            "startYearTo": startYearTo,
            "countriesOfOrigin": countriesOfOrigin,
            "spokenLanguages": spokenLanguages,
            "sortOrder": sortOrder,
            "sortField": sortField
        }
        # Remove None values
        filtered_params = {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items() if v is not None}
        query_string = urllib.parse.urlencode(filtered_params, doseq=True)
        url = f"/api/imdb/search?{query_string}"

        res = self.gateway.request(
            host=self.IMDB_API_HOST,
            method="GET",
            url=url
        )
        data = res.read()
        return json.loads(data)
    
    def search_movie_ai(self, query: str):
        # Extract parameters from user query using LLM
        search_system_prompt = (
            "Extract the following parameters from the user's movie search query. "
            "Return a valid JSON with only non-null values and no json marker. "
            "Possible keys: originalTitle, originalTitleAutocomplete, primaryTitle, primaryTitleAutocomplete, "
            "type, genre, genres, isAdult, averageRatingFrom, averageRatingTo, numVotesFrom, numVotesTo, rows, "
            "startYearFrom, startYearTo, countriesOfOrigin, spokenLanguages, sortOrder, sortField. "
            "Note genre takes a single enum value as title cased string; "
            "Parameter genres must be a array; Parameter isAdult must be a boolean; "
            "Parameter countriesOfOrigin must be a valid 'ISO 3166-1 alpha-2' country code; "
            "Parameter 'spokenLanguages' must be a valid 'ISO 639-1' language code"
            "parameter 'type' is an enum with values: 'movie', 'tvSeries', 'tvMiniSeries'; set this parameter explicitly unless the user query is ambiguous. "
            "Example output: {{'genre': 'Romance', 'spokenLanguages': 'Hi', 'startYearFrom': 1970, 'startYearTo': 1979, 'averageRatingFrom': 7}}"
        )
        search_user_query = f"User query: {query}\nExtract parameters as described."
        llm = LLM()
        params = llm.query_json(system_prompt=search_system_prompt, user_query=search_user_query)

        params["rows"] = 10
        params["sortField"] = "averageRating"
        params["sortOrder"] = "desc"
        print("Extracted parameters:", params)

        # Use RapidAPI to search for movies with the extracted parameters
        results = self.search_movie_advanced(**params)

        # Summarize the results using LLM using llm.query()
        summarize_system_prompt = (
            "You are a movie expert. Summarize the following movie search results in a concise and informative manner "
            "relevant to the exact user query based on the movie search results. "
            "Include additional details as appropriate to user query."
            # "Include the movie title, description, year, and average rating for each movie as additional details even if not asked."
        )
        input={
            "query": query,
            "movie_search_results_json": json.dumps(results, indent=2)
        }
        summarize_user_query = "Summarize the results. User query: {query} \n Movie search results: ```json\n{movie_search_results_json}\n``` "
        summary = llm.query(system_prompt=summarize_system_prompt, user_query=summarize_user_query, input=input)
        return summary, results