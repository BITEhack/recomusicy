import requests

# use this link in web http://127.0.0.1:8000/docs
url = 'http://127.0.0.1:8000/recommend?user_id=22'

files = {'user_id': 22}

response = requests.get(url, files=files)

print(response.text)
