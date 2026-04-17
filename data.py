# Data Sources: API Files Collections
# Formats: JSON CSV XML

# Example of a JSON file
import json

data = json.loads('{"name": "John", "age": 30, "city": "New York"}')
print("JSON Data:")
print(data)  # Print the entire JSON data
print("Name:", data["name"])  # Print the name

import requests

response = requests.get("https://dummyjson.com/products")

if response.status_code == 200:
    data = response.json()
    print(f"Price: {data["products"][0]['price']}$")  # Print the first product
else:
    print(f"Error: {response.status_code}")
    print("Failed to retrieve data.")

response = requests.get("https://google.com/")

if response.status_code == 200:
    print("Google is up and running!")
    print(f"Response time: {response.elapsed.total_seconds()} seconds")
    print(f"Response content: {response.content[:100]}...")  # Print the first 100 characters of the response

# Example of a CSV file
import pandas as pd

df = pd.read_csv("./assets/products.csv", delimiter=",", quotechar='"')
print("CSV Data:")
print(df.head())  # Print the first few rows of the DataFrame

print("CSV Data Types:")
print(df.dtypes)  # Print the data types of each column

print("Price:", df["Price"][0], "$")  # Print the price of the first product

# Example of an XML file
from bs4 import BeautifulSoup

root = BeautifulSoup("""
    <root>
        <name>John</name>
        <age>30</age>
    </root>
""", "xml")

print("XML Data:")
print(root.prettify())  # Print the entire XML data

print("Name:", root.find("name").text)  # Print the name

# Example of a HTML content
html = "<html><body><h1>Hello, world!</h1></body></html>"
soup = BeautifulSoup(html, "lxml")  # or "html.parser"

print(soup.h1.text)  # Output: Hello, world!
