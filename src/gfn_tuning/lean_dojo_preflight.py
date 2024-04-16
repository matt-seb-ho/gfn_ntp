import os

if "GITHUB_ACCESS_TOKEN" not in os.environ:
    from dotenv import load_dotenv
    load_dotenv("/home/matthewho/.env")
