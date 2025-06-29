import os
from dotenv import load_dotenv
from askverse_mini.api.rapid_api import RapidAPI
from askverse_mini.ai.open_api_agent import OpenAPIAgent

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME")

def test_search_name():
    rapid_api = RapidAPI()
    query = input("Enter the movie name: ")
    results = rapid_api.search_movie_name(query)
    print("Here are the movies I found:")
    movies = [f"{movie['originalTitle']}" for movie in results]
    print(movies, sep="\n")

def test_weather():
    rapid_api = RapidAPI()
    city = input("Enter the city name: ")
    days = input("Enter the number of days for the weather forecast (default is 3): ") or 3
    results = rapid_api.get_weather(city, days)
    print(f"Weather forecast for {city}:")
    for day in results['forecast']['forecastday']:
        date = day['date']
        condition = day['day']['condition']['text']
        max_temp = day['day']['maxtemp_c']
        min_temp = day['day']['mintemp_c']
        print(f"{date}: {condition}, Max Temp: {max_temp}°C, Min Temp: {min_temp}°C")

def test_search_movie_ai():
    query = input("Describe your movie search (e.g., 'Find Hindi romance movies from 1970s with rating above 7'): ")
    rapid_api = RapidAPI()
    summary, _ = rapid_api.search_movie_ai(query)
    print(f"AI Summary: {summary}")

def test_open_api_agent():
    agent = OpenAPIAgent()
    user_query = input("Enter your query for the OpenAPI agent: ")
    plan = agent.plan(user_query)
    print("Planned API calls:")
    for action in plan:
        print(action)

if __name__ == "__main__":
    load_dotenv()
    # test_weather()
    # test_search_name()
    # test_search_movie_ai()
    test_open_api_agent()