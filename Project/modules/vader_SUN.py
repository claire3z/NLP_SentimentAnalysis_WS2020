"Owner: Claire SUN"

''"""
Created by: Claire Z. Sun
Date: 2021.02.08
''"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def vader(x):
    score = analyzer.polarity_scores(x)['compound']
    # Compound score value range to corresponding sentiments according to https://github.com/cjhutto/vaderSentiment
    if score > 0.05:
        #sentiment = 'positive'
        return 2
    elif score < -0.05:
        #sentiment = 'negative'
        return 0
    else:
        #sentiment = 'neutral'
        return 1

