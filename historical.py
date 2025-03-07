import requests
from spot import *
from config import *
import pprint
import praw
from datetime import datetime, timezone
import pandas as pd
import re
from collections import defaultdict



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


def get_posts(ticker):
    ticker_count = {}
    subreddits = [
    "wallstreetbets", "finance", "investing", "stocks", "StockMarket", 
    "SecurityAnalysis", "options", "daytrading", "pennystocks",
    "algotrading", "dividends", "stockpicks", "ValueInvesting", 
    "robinhood", "stocktwits", "weedstocks", "cryptocurrencies"
]
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.search(ticker, sort='top', limit=10000):
        #for post in subreddit.top(time_filter='all', limit=None):
            post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d')
            post_id = post.id
            title = post.title
            title_split = title.split()
            num_comments = post.num_comments
            upvotes = post.score
            for token in title_split:
                if token.startswith('$'):
                    token = token[1:]
                if re.sub(r"[^\w]", "", token) == ticker:
                    if token not in ticker_count:
                        ticker_count[token] = {"Ticker": token, "Title": [title], "Post_Date": [post_date], "Upvotes": [upvotes], "Num_Comments": [num_comments], "Num_Mentions": 1}
                    else:
                        ticker_count[token]["Title"].append(title)
                        ticker_count[token]["Post_Date"].append(post_date)
                        ticker_count[token]["Upvotes"].append(upvotes)
                        ticker_count[token]["Num_Comments"].append(num_comments)
                        ticker_count[token]["Num_Mentions"] += 1


    formatted_data = []
    for symbol, data in ticker_count.items():
        for i in range(len(data["Title"])):  # Iterate over all values in lists
            if data["Upvotes"][i] > 0:
                formatted_data.append({
                    "Ticker": symbol,
                    "Title": data["Title"][i],
                    "Post_Date": data["Post_Date"][i],
                    "Upvotes": data["Upvotes"][i],
                    "Num_Comments": data["Num_Comments"][i],
                    "Num_Mentions": data["Num_Mentions"]
                    
                })

    df = pd.DataFrame(formatted_data)
    df.to_csv(f'{ticker}_reddit_data.csv', index=False)

    print(f"Historical data saved to{ticker}_reddit_data.csv!")



def get_hype():
    ticker_count = defaultdict(lambda: defaultdict(lambda: {
        "Total_Mentions": 0,
        "Total_Upvotes": 0,
        "Total_Comments": 0,
        "Post_IDs": [],
        "Titles": []
    }))
    subreddits = [
    "wallstreetbets", "finance", "investing", "stocks", "StockMarket", 
    "SecurityAnalysis", "options", "daytrading", "pennystocks",
    "algotrading", "dividends", "stockpicks", "ValueInvesting", 
    "robinhood", "stocktwits", "weedstocks", "cryptocurrencies"
]
    
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        print(f"Scraping: {sub}")
        for post in subreddit.hot(limit = 10000):
            post_title = post.title
            post_id = post.id
            num_comments = post.num_comments
            upvotes = post.score
            post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d')
            split_title = post_title.split()
            for token in split_title:
                clean_token = re.sub(r"[^\w]", "", token).upper() 
                if clean_token in watchlist:
                    ticker_data = ticker_count[post_date][clean_token]

                    ticker_data["Titles"].append(post_title)
                    ticker_data["Post_IDs"].append(post_id)
                    ticker_data["Total_Mentions"] += 1
                    ticker_data["Total_Upvotes"] += upvotes
                    ticker_data["Total_Comments"] += num_comments

        # Convert to DataFrame
    all_data = []
    for date, tickers in ticker_count.items():
        for ticker, data in tickers.items():
            all_data.append([
                date, ticker, data["Total_Mentions"], data["Total_Upvotes"],
                data["Total_Comments"], ", ".join(data["Post_IDs"]), ", ".join(data["Titles"])
            ])

    df = pd.DataFrame(all_data, columns=["Post_Date", "Ticker", "Total_Mentions", "Total_Upvotes", "Total_Comments", "Post_IDs", "Titles"])
    df.to_csv("reddit_hype.csv", index=False)

    print(f"Hype data saved to reddit_hype.csv!")



def main():
    try:
        get_posts("GME")
        get_hype()
    except KeyboardInterrupt:
        print("Exiting gracefully...")
if __name__ == '__main__':
    main()