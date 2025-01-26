from dotenv import load_dotenv
import os

class Config:

    load_dotenv()

    PUB_KEY = os.getenv('PUB_KEY')
    PRIV_KEY = os.getenv('PRIV_KEY')
    DB_URI = os.getenv('DB_URI')