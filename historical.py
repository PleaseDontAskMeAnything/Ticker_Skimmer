import requests
from spot import *
from config import *
import pprint
import praw
from datetime import datetime, timezone
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent="python:ticker_skimmer:v1.0",
    read_only=True
)

pp = pprint.PrettyPrinter(indent=1)


def get_historic_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'

    r = requests.get(url)

    data = r.json()

    return data



def get_posts():
    analyzer = SentimentIntensityAnalyzer()
    subreddit = reddit.subreddit('wallstreetbets')

    for post in subreddit.hot(limit=2000):
        tickers_found = []
        post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        #print(f"Title: {post.title}")
        title = post.title
        title_split = title.split()
        sentiment_score = analyzer.polarity_scores(title)["compound"]
        upvotes = post.score
        for token in title_split:
            if token.startswith('$'):
                token = token[1:]
            if token in watchlist:
                tickers_found.append(token)
                if token not in ticker_count:
                    ticker_count[token] = {"Ticker": token, "Title": [title], "Post_Date": post_date, "Mentions": 1, "Upvotes": upvotes, "Likes_per_mention": 0, "Sentiment_Analysis": [sentiment_score]}
                else:
                    ticker_count[token]["Mentions"] += 1
                    ticker_count[token]["Title"].append(title)
                    ticker_count[token]["Sentiment_Analysis"].append(sentiment_score)

    for ticker in ticker_count:
        mentions = ticker_count[ticker]["Mentions"]
        if ticker_count[ticker]["Upvotes"] > 0:
            ticker_count[ticker]["Likes_per_mention"] = ticker_count[ticker]["Upvotes"]/mentions

        
    df = pd.DataFrame.from_dict(ticker_count, orient="index") 
    df.to_csv('reddit_data.csv', index=False)

        # print(post_date)
        



def main():
    try:
        get_posts()
    except KeyboardInterrupt:
        print("Exiting gracefully...")


if __name__ == '__main__':
    main()