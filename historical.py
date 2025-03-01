import requests
from spot import *
import config
import pprint
import praw
from datetime import datetime, timezone

reddit = praw.Reddit(
    client_id=CLIENT_ID,  # Found at the top of your app settings
    client_secret=CLIENT_SECRET,  # The secret you see in the image
    #redirect_uri="http://localhost:8080",
    user_agent="python:ticker_skimmer:v1.0 (by /u/larrypall)",  # A short description (e.g., "TickerSkimmer bot")
    # username = "larrypall",
    # password = "archie25",
    read_only=True
)

pp = pprint.PrettyPrinter(indent=1)
symbol = 'AAPL'

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'

r = requests.get(url)

data = r.json()


def get_posts():
    subreddit = reddit.subreddit('wallstreetbets')

    for post in subreddit.hot(limit=5):
        post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Title: {post.title}")
        print(f"Body: {post.selftext}")
        print(post_date)
        



def main():
    get_posts()


if __name__ == '__main__':
    main()