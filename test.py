from dotenv import load_dotenv
from askverse_mini.rapid_api import RapidAPI

def main():
    rapid_api = RapidAPI()
    query = input("Enter the movie name: ")
    results = rapid_api.search_movie(query)
    print("Here are the movies I found:")
    movies = [f"{movie['originalTitle']}" for movie in results]
    print(movies, sep="\n")

if __name__ == "__main__":
    load_dotenv()
    main()