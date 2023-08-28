from flask import Flask, Response, request
import requests

TOKEN = ''
NGROK_URL = ''
CHAT_ID = ''

TELEGRAM_INIT_WEBHOOK_URL = 'https://api.telegram.org/bot{}/setWebhook?url={}/message'.format(TOKEN, NGROK_URL)

# Create a Flask app
app = Flask(__name__)


# Create a route that returns a message
@app.route('/sanity')
def sanity():
    return "Server is running"


@app.route('/message', methods=["POST"])
def handle_message():
    msg = request.form.get('text')
    msg = msg if msg else 'got it'
    print(msg)
    res = requests.get("https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(TOKEN, CHAT_ID, msg))
    return Response("success")


if __name__ == '__main__':
    requests.get(TELEGRAM_INIT_WEBHOOK_URL)
    app.run(port=5002)
