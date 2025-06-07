import os
import json
import http.client
import urllib.parse

class RapidAPI:
    def __init__(self):
        self.RAPID_API_KEY = os.getenv("RAPID_API_KEY")
        self.IMDB_API_HOST = os.getenv("IMDB_API_HOST")
        self.HEADERS = {
            'x-rapidapi-key': self.RAPID_API_KEY,
            'x-rapidapi-host': self.IMDB_API_HOST
        }

    def search_movie(self, query):
        query = urllib.parse.quote(query, safe='')
        conn = http.client.HTTPSConnection(self.IMDB_API_HOST)
        conn.request("GET", f"/imdb/autocomplete?query={query}", headers=self.HEADERS)

        res = conn.getresponse()
        data = res.read()
        return json.loads(data)
