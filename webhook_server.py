from flask import Flask, request
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = "NooranIshtiaq/i222010_Assignment4" 

@app.route('/webhook', methods=['POST'])
def webhook():
    print("🚨 Alert received from Prometheus!")

    url = f"https://api.github.com/repos/{REPO}/dispatches"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {
        "event_type": "retrain_trigger"
    }

    response = requests.post(url, headers=headers, json=data)
    print("GitHub response:", response.status_code)

    return "OK", 200

app.run(port=5001)