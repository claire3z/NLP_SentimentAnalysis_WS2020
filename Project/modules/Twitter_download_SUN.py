"Owner: Claire SUN"

''"""
Created by: Claire Z. Sun
Date: 2021.02.12
''"""

# Installation: $pip install tweepy
# #Successfully installed tweepy-3.9.0
# Documentaton: https://developer.twitter.com/en/docs/twitter-api;
#               http://docs.tweepy.org/en/latest/index.html
# Important Note:
### The standard search API is focused on relevance and not completeness. This means that some Tweets and users may be missing from search results. For completeness, premium (USD149/month) or enterprise search APIs.
### The recent search endpoint returns Tweets from the last 7 days that match a search query.
### At the Basic access level, you will be limited to receive 500,000 Tweets per month per project from either the filtered stream or search Tweets endpoints. For example, if you consumed 200,000 Tweets with filtered stream, you will be able to receive an additional 300,000 Tweets from either filtered stream or search Tweets. Once you have used up this allotment, you will need to wait until the next monthly period begins, which is set to the day that your developer account was approved.

import tweepy
import csv
import time
import pandas as pd

auth = tweepy.OAuthHandler(consumer_key="CSdkmSZZGtFvwxvwKDJbJJv9e", consumer_secret="tVtljpwybEUtDHSHnosiXjpcMPqZg7GHi5yi7YNj5v4yJx1hTN")
api = tweepy.API(auth, wait_on_rate_limit=True)

# The most common request limit interval is fifteen minutes. If an endpoint has a rate limit of 900 requests/15-minutes, then up to 900 requests over any 15-minute interval is allowed.
def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)


def Download_Tweets(save_path, search_term, since, until):

    with open(save_path, 'a') as csvFile:
        csvWriter = csv.writer(csvFile)
        tweets = tweepy.Cursor(api.search, q=search_term, lang="en", count=100,
                               tweet_mode='extended', since = since, until = until).items()  # default last 7 days; optional: since='2020-12-01', until="2020-12-02",
        try:
            for tweet in limit_handled(tweets):
                csvWriter.writerow([
                    tweet.created_at,  # date (ISO 8601)	Creation time of the Tweet.
                    tweet.full_text.encode('utf-8'),
                    tweet.id_str,      # string	Unique identifier of this Tweet. This is returned as a string in order to avoid complications with languages and tools that cannot handle large integers.
                    #tweet.in_reply_to_status_id_str,
                    # If this Tweet is a Reply, indicates the user ID of the parent Tweet's author. This is returned as a string in order to avoid complications with languages and tools that cannot handle large integers.
                    #tweet.retweet_count,
                    #tweet.favorite_count,
                    #tweet.is_quote_status,
                    #tweet.user.id_str,
                    #tweet.user.screen_name,
                    #tweet.user.verified,
                    #tweet.user.created_at,  # The UTC datetime that the user account was created on Twitter.
                    #tweet.user.followers_count,  # The number of followers this account currently has
                    #tweet.user.friends_count,  # The number of users this account is following (AKA their “followings”)
                    #tweet.user.statuses_count,  # The number of Tweets (including retweets) issued by the user.
                    #tweet.user.favourites_count,  # The number of Tweets this user has liked in the account’s lifetime.
                    #tweet.user.listed_count,  # The number of public lists that this user is a member of.
                ])
        except(RuntimeError):
            csvFile.close()
