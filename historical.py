import requests
from spot import *
from config import *
import pprint
import praw
from datetime import datetime, timezone

reddit = praw.Reddit(
    client_id=CLIENT_ID,  # Found at the top of your app settings
    client_secret=CLIENT_SECRET,  # The secret you see in the image
    user_agent="python:ticker_skimmer:v1.0",  # A short description (e.g., "TickerSkimmer bot")
    read_only=True
)

pp = pprint.PrettyPrinter(indent=1)
symbol = 'AAPL'


def get_historic_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'

    r = requests.get(url)

    data = r.json()



def get_posts():
    subreddit = reddit.subreddit('wallstreetbets')

    for post in subreddit.top(limit=100):
        post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        #print(f"Title: {post.title}")
        title = post.title
        title = title.split()
        for token in title:
            if token.startswith('$'):
                token = token[1:]
            if token in watchlist:
                if token not in ticker_count:
                    ticker_count[token] = 1
                else:
                    ticker_count[token] += 1
        
    print(ticker_count)

                #get_historic_data(token)
        # if post.selftext:
        #     print(f"Body: {post.selftext}")
        # else:
        #     print("No text based body")

        # print(post_date)
        



def main():
    get_posts()


if __name__ == '__main__':
    main()