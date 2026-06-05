import requests
r = requests.head("http://aisdata.ais.dk/2021/aisdk-2021-12-13.csv")
print("Status:", r.status_code)
print("Content-Length:", r.headers.get("content-length"))