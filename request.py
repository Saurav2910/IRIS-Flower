import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sepal-length':3.5, 'sepal-width':2.5, 'petal-length':2.8, 'petal-width':1.8})

print(r.json())