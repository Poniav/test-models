"""
Config file for the application
Init the basic settings 
"""

from dotenv import load_dotenv
import os

class Config:

    load_dotenv()

    # Load the environment variables
    PUB_KEY = os.getenv('PUB_KEY')
    PRIV_KEY = os.getenv('PRIV_KEY')
    DB_URI = os.getenv('DB_URI')

    # Path to the resources folder
    RESOURCES_PATH = os.path.abspath("resources")