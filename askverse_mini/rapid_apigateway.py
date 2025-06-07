import os
import http.client

class RapidAPIGateway:
    def __init__(self):
        self.rapid_api_key = os.getenv("RAPID_API_KEY")
        if not self.rapid_api_key:
            raise ValueError("RAPID_API_KEY environment variable is not set")

    def request(self, host, method, url, body=None, headers=None, *, encode_chunked=False):
        if not host.endswith('.rapidapi.com'):
            raise ValueError("Host must end with '.rapidapi.com'")
        
        # Prepare headers and inject RapidAPI credentials
        headers = headers.copy() if headers else {}
        headers['x-rapidapi-key'] = self.rapid_api_key
        headers['x-rapidapi-host'] = host

        conn = http.client.HTTPSConnection(host)
        conn.request(method, url, body=body, headers=headers, encode_chunked=encode_chunked)
        return conn.getresponse()