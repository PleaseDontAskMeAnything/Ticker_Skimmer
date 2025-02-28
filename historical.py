import requests
from spot import *
import pprint
import praw

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",  # Found at the top of your app settings
    client_secret="YOUR_SECRET",  # The secret you see in the image
    user_agent="YOUR_USER_AGENT",  # A short description (e.g., "TickerSkimmer bot")
)

pp = pprint.PrettyPrinter(indent=1)
symbol = 'AAPL'

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'

r = requests.get(url)

data = r.json()

pp.pprint(data)



def main():
    print(reddit.read_only)


if __name__ == '__main__':
    main()