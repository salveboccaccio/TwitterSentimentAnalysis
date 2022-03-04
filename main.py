#importing stuff
import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.cbook

#Removing Pandas and MatPlotLib warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter(action='ignore', category=FutureWarning)

#To see full dataframe without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#remove emojis from text
def no_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF"  
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, "", text)

#Asking for ticker and how many tweets to look at
ticker = input('enter ticker: ')
no_of_tweets = int(input('how many tweets: ')) - 1

tweets_list = []

# Using TwitterSearchScraper to scrape data and append tweets to list, while also cleaning up text
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(ticker).get_items()):
    if i>no_of_tweets:
        break
    clean_tweet = tweet.content.replace("\n", "")
    clean_tweet = clean_tweet.replace("/", "")
    clean_tweet = clean_tweet.replace("&amp", "")
    clean_tweet = no_emoji(clean_tweet)
    tweets_list.append(clean_tweet)

textblob_sentiment = []

#doing sentiment analysis and putting into dataframe
for s in tweets_list:
    analysis = TextBlob(s)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    overall_sentiment = ''
    if analysis.sentiment[0]>0:
        overall_sentiment = 'positive'
    elif analysis.sentiment[0]<0:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'
    textblob_sentiment.append([s, float(polarity), subjectivity, overall_sentiment])

tweets_dataframe = pd.DataFrame(textblob_sentiment, columns = ['tweets', 'Polarity', 'Subjectivity', 'Sentiment'])

tweets_df = pd.DataFrame(tweets_dataframe)
tweets_df.drop_duplicates()

positive_tweets = pd.DataFrame()
negative_tweets = pd.DataFrame()
neutral_tweets = pd.DataFrame()

#creating positive/neutral/negative dataframe
positive_tweets.append(tweets_df.loc[tweets_df['Polarity'] > 0.00])
negative_tweets.append(tweets_df.loc[tweets_df['Polarity'] < 0.00])
neutral_tweets.append(tweets_df.loc[tweets_df['Polarity'] == 0.00])

positive_tweets = positive_tweets.append(tweets_df.loc[tweets_df['Polarity'] > 0.00])
negative_tweets = negative_tweets.append(tweets_df.loc[tweets_df['Polarity'] < 0.00])
neutral_tweets = neutral_tweets.append(tweets_df.loc[tweets_df['Polarity'] == 0.00])

#printing basic information about dataframe
print("Analysis of sentiment for " + ticker + " over last", no_of_tweets + 1, "tweets:")
print("Tweets with positive sentiment:", len(positive_tweets), '(', 100 * (len(positive_tweets)/(no_of_tweets + 1)), '%)')
print("Tweets with neutral sentiment:", len(neutral_tweets), '(', 100 * (len(neutral_tweets)/(no_of_tweets + 1)), '%)')
print("Tweets with negative sentiment:", len(negative_tweets), '(', 100 * (len(negative_tweets)/(no_of_tweets + 1)), '%)')

#plotting pie chart and bar graph
number_of_tweets = [len(positive_tweets), len(neutral_tweets), len(negative_tweets)]
graph_sentiment = ['Positive', 'Neutral', 'Negative']
graph_colors = ['green', 'gray', 'red']

#bar graph
plt.subplot(1,2,1)
plt.bar(graph_sentiment,number_of_tweets, color=graph_colors)
plt.title('Sentiment Vs Number of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')

#pie chart
plt.subplot(1,2,2)
y = np.array(number_of_tweets)
plt.pie(y, labels=graph_sentiment, colors=graph_colors)
plt.legend()

plt.show()


