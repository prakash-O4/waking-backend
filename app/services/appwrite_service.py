import os
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.id import ID


def getClient():
    print(os.getenv('API_END_POINT'))
    print("test test")
    client = Client()
    client.set_endpoint(os.getenv('API_END_POINT'))
    client.set_project(os.getenv('PROJECT_ID'))
    client.set_key(os.getenv('API_KEY'))
    return client