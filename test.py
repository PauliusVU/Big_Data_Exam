import requests, zipfile, io, pandas as pd

url = "http://aisdata.ais.dk/2021/aisdk-2021-12.zip"
print("Downloading...")
r = requests.get(url, stream=True)
print(f"Status: {r.status_code}")

with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    print("Files inside zip:", z.namelist())
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, nrows=10)
        print("\nColumns:", df.columns.tolist())
        print("\nSample rows:")
        print(df.head(10))
        print("\nDtypes:")
        print(df.dtypes)