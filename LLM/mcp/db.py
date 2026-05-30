import os

import pytds
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return pytds.connect(
        server=os.getenv("DB_SERVER"),
        port=int(os.getenv("DB_PORT", "1433")),
        database=os.getenv("DB_DATABASE"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        timeout=15,
        login_timeout=15,
    )