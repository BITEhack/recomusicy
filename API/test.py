import requests

url = 'http://127.0.0.1:8000/classification?user_id=10&lyrics=adg'

# files = {'image': open('example.jpg', 'rb'),
#          'user_id': 10}

files = {'lyrics': 'adg',
         'user_id': 10}

response = requests.post(url, files=files)

print(response.text)
