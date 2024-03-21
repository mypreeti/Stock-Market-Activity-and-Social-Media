# Stock-Market-Activity-and-Social-Media
Stock Market Activity and Its Relationship with Social Media

Title: Unraveling the Social Media Stock Market Nexus: Exploring the Influence of Reddit and Twitter on Tesla, Apple, and Zoom

In the ever-evolving landscape of financial markets, the intersection of social media and stock prices has become increasingly pronounced. A recent tweet by Elon Musk, the maverick billionaire and CEO of Tesla Motors, merely captioned "Gamestonk!!" alongside a link to a Reddit thread on WallStreetBets, sent shockwaves through the market, notably driving up the valuation of GameStop. This event underscored the potential influence of social media discourse on stock movements, prompting a deeper inquiry into its dynamics.

Motivated by this phenomenon, our research embarked on a quest to dissect the relationship between social media chatter and stock performance, focusing on three tech giants: Apple, Tesla, and Zoom. Our investigation spanned from January 1, 2020, to January 1, 2021, harnessing the power of web scraping to gather mentions of these stocks across Reddit and Twitter. Our objective was clear: to discern whether the frequency of discussions on these platforms correlated with fluctuations in stock prices.

The preliminary findings of my study unveiled a nuanced correlation between the volume of Reddit and Twitter posts and the average weekly closing prices of TSLA, AAPL, and ZM. However, it is crucial to note that while this correlation leaned toward the positive spectrum, its magnitude varied from weak to moderate. 

Despite the intriguing connections unearthed, our analysis encountered a formidable hurdle in constructing predictive models: the lack of homoscedasticity within the dataset. This statistical concept, which pertains to the equal variance of data points, posed a significant challenge in crafting robust predictive frameworks.

While our findings shed light on the influence of social media on stock prices, they also underscore the complexity of predicting market movements solely based on online discourse. As we navigate this dynamic interplay between social media dynamics and financial markets, further exploration is needed to understand its implications fully.

Is there a relationship between the frequency of mentions of Tesla, Apple, and Zoom on Reddit and Twitter and the change in their respective average weekly closing prices from 01/01/2020 to 01/01/2021?

Dataset(s)
We are using multiple datasets, particular datasets containing twitter mentions, stock information and reddit submission and comments. The datasets will eventually be joined via the week, as they are time series data.

Cumulative Dataset Name: StockTwitter
Total number of observations: 1945357
Dataset Name: AAPLTwitter
Link to the dataset: https://drive.google.com/file/d/1NVk8jlPGvm7BuqoCIlGWFUF56wYhTW4Z/view?usp=sharing
Number of observations: 502673
Dataset Name: TSLATwitter
Link to the dataset: https://drive.google.com/file/d/1NVk8jlPGvm7BuqoCIlGWFUF56wYhTW4Z/view?usp=sharing
Number of observations: 1291646
Dataset Name: ZMTwitter
Link to the dataset: https://drive.google.com/file/d/1NVk8jlPGvm7BuqoCIlGWFUF56wYhTW4Z/view?usp=sharing
Number of observations: 151038
The above datasets of AAPLTwitter, TSLATwitter, and ZMTwitter are under a cumulative dataset name of StockTwitter. The dataset was created by group006 using the snscrape module. The tweets with relevant stock symbols (TSLA, AAPL, and ZM) from January 1st, 2020, to January 1st, 2021 were gathered. Since we measure the frequency of tweets, the user information and tweet content outside of the stock symbols is irrelevant, and we decided not to collect that information. Specifically, we collect the date and tweet made that contains the relevant stock symbols (TSLA, AAPL, and ZM) along with the unique tweet ID.
Dataset Name: StockPrices
Link to the dataset:https://drive.google.com/drive/folders/1jH-z8dSNCXbYUra6RV6Jqjm12jRjbOae?usp=sharing
Number of observations: 5,313
The above dataset was created by group006 using the Yahoo Finance package to retrieve historic market data from Yahoo Finance. The open, high, low, close, volume, dividends, and stock split values of relevant stock symbols (TSLA, AAPL, and ZM) from January 1st, 2020, to January 1st, 2021 were gathered. Since we are measuring the price and volume of the relevant stocks, the dividends and stock split values were found unessential for this project and thus dropped. The values that were gathered are over a five-day period.
Dataset Name: StockReddit
Link to the dataset: https://drive.google.com/drive/folders/1-sNqjfy8jUDjrC8xH8NHx19ci24-cCFK?usp=sharing
Number of observations: 511001 (Comments) + 36476 (Submissions)
The above dataset was created by group006 using the PMAW - a wrapper for the pushshift.io Reddit API. The Pushshift runs an archive of all Reddit comments and submissions from January 1st, 2020 to January 1st, 2021, where all mentions of the relevant stock symbols (TSLA, AAPL, and ZM) were gathered. All of the comments and submissions with matching stock symbols data were scraped and placed into .csv files hosted on OneDrive. The data was then subsequently cleaned to remove any potential personally-identifying information and the data that might not be useful and resaved into new .csv files.


Set up

import sys
!{sys.executable} -m pip install pmaw
!{sys.executable} -m pip install git+https://github.com/JustAnotherArchivist/snscrape.git

#Import reddit scraping package
from pmaw import PushshiftAPI

#Import $TSLA, $AAPL, and $ZM Stock Prices and Volume
#Install stock market data package, details of package below
!{sys.executable} -m pip install --user yfinance --upgrade --no-cache-dir
#This imported package downloads historical stock market data from Yahoo! finance, a reliable source of stock market info
import yfinance as yf

#Import other useful packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# Suppress the warnings
import warnings 
warnings.filterwarnings('ignore')

#Define the ticker symbol for each stock
tesla_stock = 'TSLA'
apple_stock = 'AAPL'
zoom_stock = 'ZM'

#Gather data on each ticker
tesla_data = yf.Ticker('TSLA')
apple_data = yf.Ticker('AAPL')
zoom_data = yf.Ticker('ZM')

#Gather the historical stock market values for each ticker\n",
tesla_df = tesla_data.history(period='5d', start='2020-1-1', end='2021-1-1')
apple_df = apple_data.history(period='5d', start='2020-1-1', end='2021-1-1')
zoom_df = zoom_data.history(period='5d', start='2020-1-1', end='2021-1-1')

#Transform gathered data and dataframes into csv's to be linked and read
tesla_df.to_csv('tesla_df.csv')
apple_df.to_csv('apple_df.csv')
zoom_df.to_csv('zoom_df.csv')

## Load the reddit comment csvs into dataframes
#Reddit comments
reddit_comments_TSLA_df = pd.read_csv(f'./data/reddit_comments_TSLA.csv')
reddit_comments_AAPL_df = pd.read_csv(f'./data/reddit_comments_AAPL.csv')
reddit_comments_ZM_df = pd.read_csv(f'./data/reddit_comments_ZM.csv')
reddit_comments_TSLA_AAPL_df = reddit_comments_TSLA_df.merge(reddit_comments_AAPL_df, how = 'outer')
reddit_comments_all_df = reddit_comments_TSLA_AAPL_df.merge(reddit_comments_ZM_df, how = 'outer')
reddit_comments_all_df = reddit_comments_all_df.drop_duplicates().reset_index()
print(reddit_comments_all_df.columns)
print(reddit_comments_all_df.shape)

#import Twitter Data
TSLA_Tw = pd.read_csv("TSLATwitter.csv")
AAPL_Tw = pd.read_csv("AAPLTwitter.csv")
ZM_Tw = pd.read_csv("ZMTwitter.csv")

#Stock Price and Volume Data Cleaning
tesla_df = pd.read_csv("tesla_df.csv")
apple_df = pd.read_csv("apple_df.csv")
zoom_df = pd.read_csv("zoom_df.csv")
#Check how large our dataframes are
a=tesla_df.shape
b=apple_df.shape
c=zoom_df.shape
print(a,'\n')
print(b,'\n')
print(c)
tesla_df.head()

#Check what columns we have in our DataFrames
a=tesla_df.columns
b=apple_df.columns
c=zoom_df.columns
print(a,'\n')
print(b,'\n')
print(c)

#Check the datatypes of our variables
a=tesla_df.dtypes
b=apple_df.dtypes
c=zoom_df.dtypes
print(a,'\n')
print(b,'\n')
print(c)

#Get descriptive statistics of all numerical columns
a=tesla_df.describe()
b=apple_df.describe()
c=zoom_df.describe()
print(a,'\n')
print(b,'\n')
print(c)

#Specify which columns to include, since we are only interested analyzing the average close value across each week,
#columns besides 'Close' and 'Volume' are dropped; 'Volume' acts as a confounding variable towards 'Close'
tesla_df = tesla_df[['Date','Close', 'Volume']]
a=tesla_df.head()
apple_df = apple_df[['Date','Close', 'Volume']]
b=apple_df.head()
zoom_df = zoom_df[['Date','Close', 'Volume']]
c=zoom_df.head()
print(a,'\n')
print(b,'\n')
print(c)

#We should not have negative values, so check if we have any data from prices below the value of 0
a=sum(tesla_df['Close']< 0)+sum(tesla_df['Volume']< 0)
b=sum(apple_df['Close']< 0)+sum(apple_df['Volume']< 0)
c=sum(zoom_df['Close']< 0)+sum(zoom_df['Volume']< 0)
print(a,'\n')
print(b,'\n')
print(c)

#Check for missing values
a=(tesla_df['Close'].hasnans or tesla_df['Volume'].hasnans)
b=(apple_df['Close'].hasnans or apple_df['Volume'].hasnans)
c=(zoom_df['Close'].hasnans or zoom_df['Volume'].hasnans)
print(a,'\n')
print(b,'\n')
print(c)

#Drop all miscellaneous metadata
reddit_comments_all_df = reddit_comments_all_df.drop(columns=['all_awardings', 'associated_award', 'author_flair_background_color', 'author_flair_css_class',
                                                              'author_flair_richtext', 'author_flair_template_id','author_flair_text',
                                                              'author_flair_text_color','author_patreon_flair', 
                                                              'author_flair_type','collapsed_because_crowd_control','steward_reports', 
                                                              'distinguished', 'media_metadata','send_replies', 'stickied','permalink', 
                                                              'retrieved_on','author_cakeday','treatment_tags', 'edited', 'locked', 
                                                              'parent_id','top_awarded_type', 'comment_type','awarders', 'author_fullname',
                                                              'no_follow','link_id','subreddit_id','is_submitter','author_premium','index',
                                                              'author','gildings','id'])
print(reddit_comments_all_df.columns)
print(reddit_comments_all_df.shape)

#Initial datasets and shape
print(TSLA_Tw.shape)
print(AAPL_Tw.shape)
print(ZM_Tw.shape)
print(TSLA_Tw.head())
print(AAPL_Tw.head())
print(ZM_Tw.head())

#Describe data sets for each stock-twitter dataframe
print(TSLA_Tw.describe())
print(AAPL_Tw.describe())
print(ZM_Tw.describe())

#Partition dates to be within weeks
TSLA_Tw["Week"] = pd.to_datetime(TSLA_Tw['Datetime']).dt.week
AAPL_Tw["Week"] = pd.to_datetime(AAPL_Tw['Datetime']).dt.week
ZM_Tw["Week"] = pd.to_datetime(ZM_Tw['Datetime']).dt.week
print(TSLA_Tw.head())
print(AAPL_Tw.head())
print(ZM_Tw.head())

TSLA_AAPL= pd.merge(TSLA_Tw, AAPL_Tw, how='inner', left_on='Tweet ID', right_on='Tweet ID')
TSLA_ZM = pd.merge(TSLA_Tw, ZM_Tw, how='inner', left_on='Tweet ID', right_on='Tweet ID')
AAPL_ZM = pd.merge(AAPL_Tw, ZM_Tw, how='inner', left_on='Tweet ID', right_on='Tweet ID')

print("Percent of Tesla tweets that contain Apple: " + str(TSLA_AAPL.shape[0]/TSLA_Tw.shape[0]))
print("Percent of Apple tweets that contain Tesla: "+str(TSLA_AAPL.shape[0]/AAPL_Tw.shape[0]))
print("Percent of Apple tweets that contain Zoom: "+str(AAPL_ZM.shape[0]/AAPL_Tw.shape[0]))
print("Percent of Zoom tweets that contain Apple: "+str(AAPL_ZM.shape[0]/ZM_Tw.shape[0]))
print("Percent of Tesla tweets that contain Zoom: " +str(TSLA_ZM.shape[0]/TSLA_Tw.shape[0]))
print("Percent of Zoom tweets that contain Tesla: " +str(TSLA_ZM.shape[0]/ZM_Tw.shape[0]))
(TSLA_AAPL.head())

#Cast the to_datetime method on the index of dates within each dataframe to ensure the dates are in correct format
tesla_df.index = pd.to_datetime(tesla_df['Date'])
apple_df.index = pd.to_datetime(apple_df['Date'])
zoom_df.index = pd.to_datetime(zoom_df['Date'])
tesla_df = tesla_df.drop(['Date'],axis=1)
apple_df = apple_df.drop(['Date'],axis=1)
zoom_df = zoom_df.drop(['Date'],axis=1)
zoom_df

#Find the week number of each data, and create a new column containing these values
tesla_dates = pd.DatetimeIndex(tesla_df.index)
tesla_df['Week_Number_of_Year']= tesla_dates.week
apple_dates = pd.DatetimeIndex(apple_df.index)
apple_df['Week_Number_of_Year']= apple_dates.week
zoom_dates = pd.DatetimeIndex(zoom_df.index)
zoom_df['Week_Number_of_Year']= zoom_dates.week
print(tesla_df)
print(apple_df)
print(zoom_df)

#Now, the stock price data at close of market can be visiualized in terms of weeks rather than individual days
a = sns.lmplot(y='Close', x='Week_Number_of_Year', data = tesla_df, fit_reg=False)
a = plt.gca()
plt.xlabel("Week")
plt.ylabel("Close Price")
a.set_title('$TSLA Close Values in each Week from 01/01/20 - 01/01/21')

b = sns.lmplot(y='Close', x='Week_Number_of_Year', data = apple_df, fit_reg=False)
b = plt.gca()
plt.xlabel("Week")
plt.ylabel("Close Price")
b.set_title('$AAPL Close Values in each Week from 01/01/20 - 01/01/21')

c = sns.lmplot(y='Close', x='Week_Number_of_Year', data = zoom_df, fit_reg=False)
c = plt.gca()
plt.xlabel("Week")
plt.ylabel("Close Price")
c.set_title('$ZM Close Values in each Week from 01/01/20 - 01/01/21')

d = sns.lmplot(y='Volume', x='Week_Number_of_Year', data = tesla_df, fit_reg=False)
d = plt.gca()
plt.xlabel("Week")
plt.ylabel("Close Price")
d.set_title('$TSLA Volume in each Week from 01/01/20 - 01/01/21')

e = sns.lmplot(y='Volume', x='Week_Number_of_Year', data = apple_df, fit_reg=False)
e = plt.gca()
plt.xlabel("Week")
plt.ylabel("Close Price")
e.set_title('$AAPL Volume in each Week from 01/01/20 - 01/01/21')

f = sns.lmplot(y='Volume', x='Week_Number_of_Year', data = zoom_df, fit_reg=False)
f = plt.gca()
plt.xlabel("Week")
plt.ylabel("Close Price")
f.set_title('$ZM Volume in each Week from 01/01/20 - 01/01/21')

print(a,'\n')
print(b,'\n')
print(c,'\n')
print(d,'\n')
print(e,'\n')
print(f)

#social media data, specifically reddit
reddit_df = pd.read_csv("reddit_all.csv")

print(reddit_df.dtypes)
reddit_df.head()

reddit_df.describe()

reddit_df['subreddit'].unique().shape

subreddit_counts = reddit_df['subreddit'].value_counts()
print(subreddit_counts)
# We figure out how many subreddits are involved in trading
print(f"Removing subreddits with a single related post, we get {len(subreddit_counts[subreddit_counts > 1])} subreddits")
print(f"Removing subreddits with two related posts, we get {len(subreddit_counts[subreddit_counts > 2])} subreddits")
print(f"Removing subreddits with three related posts, we get {len(subreddit_counts[subreddit_counts > 3])} subreddits")
print(f"Removing subreddits with ten related posts, we get {len(subreddit_counts[subreddit_counts > 10])} subreddits")
print(f"Removing subreddits with a hundred related posts, we get {len(subreddit_counts[subreddit_counts > 100])} subreddits")

sns.countplot(data=reddit_df, y='subreddit',order= reddit_df['subreddit'].value_counts().iloc[:10].index)
plt.title("Top 10 subreddits who mention TSLA, AAPL and ZM")
plt.ylabel("Subreddit")
plt.xlabel("Number of posts/comments")
plt.xscale('log')

# Visualize with a histogram the subreddit distribution
sns.histplot(subreddit_counts.values, bins = 100)
plt.title("Post distribution of subreddits mentioning TSLA, AAPL and ZM")
plt.xlabel('Number of posts in subreddit')
plt.ylabel('Count of subreddits')
plt.yscale('log')

sns.histplot(subreddit_counts[subreddit_counts > 100].values, bins = 100)
plt.title("Truncated post distribution of Subreddits mentioning TSLA, AAPL and ZM")
plt.xlabel('Number of posts in subreddit')
plt.ylabel('Count of subreddits')
plt.yscale('log')

sns.histplot(subreddit_counts[subreddit_counts > 1000].values, bins = 100)
plt.title("Truncated post distribution of Subreddits mentioning TSLA, AAPL and ZM")
plt.xlabel('Number of posts in subreddit')
plt.ylabel('Count of subreddits')

print(f"There are {subreddit_counts.sum()} entries in the dataset, with a total of {len(subreddit_counts)} subreddits represented in the dataset")
print(f"After pruning the dataset, we have {subreddit_counts[subreddit_counts > 1000].sum()} entries left, with a total of {len(subreddit_counts[subreddit_counts > 1000])} subreddits represented in the pruned dataset")

reddit_df[['TSLA','AAPL','ZM']].mean()

reddit_df[['TSLA','AAPL','ZM']].sum()

award_counts = reddit_df['total_awards_received'].value_counts()
award_counts

reddit_df[reddit_df['total_awards_received'] > 10].head(100)

current = sns.barplot( x = award_counts[1:].index,y = award_counts[1:].values)
plt.xlabel("Number of Awards")
plt.ylabel("Number of Comments")
plt.title("Number of Awards per Comment")
current.set_yscale("log")

score_counts = reddit_df['score'].value_counts().sort_index()
score_counts
current = sns.barplot( x = score_counts.index,y = score_counts.values)
plt.xlabel("Number of Awards")
plt.ylabel("Number of Comments")
plt.title("Number of Awards per Comment")
current.set_yscale("log")
current.set_xscale("symlog") #symmetric log

print(f"There are {sum(score_counts)} comments and posts")
print(f"There are {sum(score_counts[score_counts.index == 1])} comments and posts with a score of 1")
print(f"There are {sum(score_counts[score_counts.index > 1])} comments and posts with a score greater than 1")
print(f"There are {sum(score_counts[score_counts.index < 1])} comments and posts with a score less than 1")

#Separate the reddit dataframe by stock 
TSLA_reddit = reddit_df[reddit_df["TSLA"]==True]
ZM_reddit = reddit_df[reddit_df["ZM"]==True]
AAPL_reddit = reddit_df[reddit_df["AAPL"]==True]

TSLA_redditCount = TSLA_reddit.groupby(["week"]).count()
ZM_redditCount = ZM_reddit.groupby(["week"]).count()
AAPL_redditCount = AAPL_reddit.groupby(["week"]).count()
TSLA_TCount = TSLA_Tw.groupby(["Week"]).count()
ZM_TCount = ZM_Tw.groupby(["Week"]).count()
AAPL_TCount = AAPL_Tw.groupby(["Week"]).count()
print(AAPL_redditCount.head())
print(AAPL_TCount.head())

#Extracting the counts from each dataframe
TeslaCountR = list(TSLA_redditCount["score"])
ZMCountR = list(ZM_redditCount["score"])
AAPLCountR = list(AAPL_redditCount["score"])
TSLACountTwit= list(TSLA_TCount["Datetime"])
AAPLCountTwit= list(AAPL_TCount["Datetime"])
ZMCountTwit= list(ZM_TCount["Datetime"])
Week = list(range(1,54))
TSLATR = pd.DataFrame(list(zip(TeslaCountR, TSLACountTwit)), columns = ["Mentions on Reddit", "Mentions on Twitter"],index = Week, dtype = int)
AAPLTR = pd.DataFrame(list(zip(AAPLCountR, AAPLCountTwit)), columns = [ "Mentions on Reddit", "Mentions on Twitter"],index = Week,dtype = int)
ZMTR = pd.DataFrame(list(zip(ZMCountR, ZMCountTwit)), columns = [ "Mentions on Reddit", "Mentions on Twitter"],index = Week,dtype = int)
print(TSLATR.head())
print(AAPLTR.head())
print(ZMTR.head())

TSLATR.plot(rot=0)
plt.xlabel("Week")
plt.ylabel("Number of Mentions")
plt.title("Mentions per Social Media Platform for TSLA")
AAPLTR.plot(rot=0)
plt.xlabel("Week")
plt.ylabel("Number of Mentions")
plt.title("Mentions per Social Media Platform for AAPL")
ZMTR.plot(rot=0)
plt.xlabel("Week")
plt.ylabel("Number of Mentions")
plt.title("Mentions per Social Media Platform for ZM")

TSLATR.corr()

AAPLTR.corr()

# Convert columns into mean, standard deviation, max and min by group for stock information
tesla_week_df = tesla_df.groupby('Week_Number_of_Year').agg({'Close' : ['mean', 'std'], 'Volume' : ['mean', 'std']})
apple_week_df = apple_df.groupby('Week_Number_of_Year').agg({'Close' : ['mean', 'std'], 'Volume' : ['mean', 'std']})
zoom_week_df = zoom_df.groupby('Week_Number_of_Year').agg({'Close' : ['mean', 'std'], 'Volume' : ['mean', 'std']})
# Clean and relabel the columns
new_stock_cols = ['Average Closing Price','Closing Price std','Average Volume','Volume std']
tesla_week_df.columns = new_stock_cols
apple_week_df.columns = new_stock_cols
zoom_week_df.columns = new_stock_cols

# Sanity check - see the first 5 rows
apple_week_df.head(5)

# Drop the relevant columns
TSLA_reddit = TSLA_reddit.drop(['created_utc','subreddit','body'],axis=1)
ZM_reddit = ZM_reddit.drop(['created_utc','subreddit','body'],axis=1)
AAPL_reddit = AAPL_reddit.drop(['created_utc','subreddit','body'],axis=1)
# Sanity check
TSLA_reddit.head()

# Groupby and aggregate reddit data
TSLA_reddit_agg = TSLA_reddit.groupby('week').agg({'score':['mean','std'],'total_awards_received':['sum'],'TSLA':['sum'],'AAPL':['sum'],'ZM':['sum'],'comment':['mean']})
AAPL_reddit_agg = AAPL_reddit.groupby('week').agg({'score':['mean','std'],'total_awards_received':['sum'],'TSLA':['sum'],'AAPL':['sum'],'ZM':['sum'],'comment':['mean']})
ZM_reddit_agg = ZM_reddit.groupby('week').agg({'score':['mean','std'],'total_awards_received':['sum'],'TSLA':['sum'],'AAPL':['sum'],'ZM':['sum'],'comment':['mean']})
# Relabel reddit dataframe columns
new_reddit_cols = ['score mean','score std', 'reddit awards given','TSLA reddit mentions','AAPL reddit mentions','ZM reddit mentions','comment/post ratio']
TSLA_reddit_agg.columns = new_reddit_cols
AAPL_reddit_agg.columns = new_reddit_cols
ZM_reddit_agg.columns = new_reddit_cols
# Sanity check
TSLA_reddit_agg.head(5)

# Aggregate twitter data -  We first clean unnecessary data
TSLA_TCount = TSLA_TCount.drop(['Datetime','Tweet ID'],axis =1)
AAPL_TCount = AAPL_TCount.drop(['Datetime','Tweet ID'],axis =1)
ZM_TCount = ZM_TCount.drop(['Datetime','Tweet ID'],axis =1)
# Relabel columns
new_TW_cols = ['Twitter mentions']
TSLA_TCount.columns = new_TW_cols 
AAPL_TCount.columns = new_TW_cols 
ZM_TCount.columns = new_TW_cols
ZM_TCount.head(5)

# Merge all three tables together into individual stock data
TSLA_reddit_agg2 = TSLA_reddit_agg.merge(TSLA_TCount,how='left',left_on='week',right_on = 'Week',right_index= True, sort = True)
tesla_all_df = TSLA_reddit_agg2.merge(tesla_week_df,how = 'left',left_on ='week' , right_on = 'Week_Number_of_Year',right_index= True, sort = True)
AAPL_reddit_agg2 = AAPL_reddit_agg.merge(AAPL_TCount,how='left',left_on='week',right_on = 'Week',right_index= True, sort = True)
apple_all_df = AAPL_reddit_agg2.merge(apple_week_df,how = 'left',left_on ='week' , right_on = 'Week_Number_of_Year',right_index= True, sort = True)
ZM_reddit_agg2 = ZM_reddit_agg.merge(ZM_TCount,how='left',left_on='week',right_on = 'Week',right_index= True, sort = True)
zoom_all_df = ZM_reddit_agg2.merge(zoom_week_df,how = 'left',left_on ='week' , right_on = 'Week_Number_of_Year',right_index= True, sort = True)
print(tesla_all_df.dtypes)
tesla_all_df.head(5)

print(apple_all_df.dtypes)
apple_all_df.head(5)

print(zoom_all_df.dtypes)
zoom_all_df.head(5)

fig, axes = plt.subplots(3,2,figsize=(16.5,17.5))
axes[0,0].set_title('Average Closing Price against TSLA reddit mentions')
sns.scatterplot(
    ax = axes[0,0], data=tesla_all_df, x="TSLA reddit mentions", y="Average Closing Price"
)
axes[1,0].set_title('Average Closing Price against AAPL reddit mentions')
sns.scatterplot(
    ax = axes[1,0], data=apple_all_df, x="AAPL reddit mentions", y="Average Closing Price"
)
axes[2,0].set_title('Average Closing Price against ZM reddit mentions')
sns.scatterplot(
    ax = axes[2,0], data=zoom_all_df, x="ZM reddit mentions", y="Average Closing Price"
)
axes[0,1].set_title('Average Closing Price against TSLA twitter mentions')
sns.scatterplot(
    ax = axes[0,1], data=tesla_all_df, x="Twitter mentions", y="Average Closing Price"
)
axes[1,1].set_title('Average Closing Price against AAPL twitter mentions')
sns.scatterplot(
    ax = axes[1,1], data=apple_all_df, x="Twitter mentions", y="Average Closing Price"
)
axes[2,1].set_title('Average Closing Price against ZM twitter mentions')
sns.scatterplot(
    ax = axes[2,1], data=zoom_all_df, x="Twitter mentions", y="Average Closing Price"
)

fig, axes = plt.subplots(3,2,figsize=(16.5,17.5))
axes[0,0].set_title('Average traded stock volume against TSLA reddit mentions')
sns.scatterplot(
    ax = axes[0,0], data=tesla_all_df, x="TSLA reddit mentions", y="Average Volume"
)
axes[1,0].set_title('Average traded stock volume against AAPL reddit mentions')
sns.scatterplot(
    ax = axes[1,0], data=apple_all_df, x="AAPL reddit mentions", y="Average Volume"
)
axes[2,0].set_title('Average traded stock volume against ZM reddit mentions')
sns.scatterplot(
    ax = axes[2,0], data=zoom_all_df, x="ZM reddit mentions", y="Average Volume"
)
axes[0,1].set_title('Average traded stock volume against TSLA twitter mentions')
sns.scatterplot(
    ax = axes[0,1], data=tesla_all_df, x="Twitter mentions", y="Average Volume"
)
axes[1,1].set_title('Average traded stock volume against AAPL twitter mentions')
sns.scatterplot(
    ax = axes[1,1], data=apple_all_df, x="Twitter mentions", y="Average Volume"
)
axes[2,1].set_title('Average traded stock volume against ZM twitter mentions')
sns.scatterplot(
    ax = axes[2,1], data=zoom_all_df, x="Twitter mentions", y="Average Volume"
)

#Function for finding absolute difference in stock price between weeks
tesla_cp = list(tesla_all_df["Average Closing Price"])
apple_cp = list(apple_all_df["Average Closing Price"])
zoom_cp = list(zoom_all_df["Average Closing Price"])
def diff_price(lis):
    a= []
    for i in range(len(lis)):
        if i ==0:
            a.append(0)
            continue
        else:
            diff = abs(lis[i] - lis[i-1])
            a.append(diff)
    return a
tesla_all_df["Absolute Weekly Difference"] = diff_price(tesla_cp)
apple_all_df["Absolute Weekly Difference"] = diff_price(apple_cp)
zoom_all_df["Absolute Weekly Difference"] = diff_price(zoom_cp)
tesla_all_df["Absolute Weekly Difference"].describe()

apple_all_df["Absolute Weekly Difference"].describe()

zoom_all_df["Absolute Weekly Difference"].describe()

fig, (ax1,ax3) = plt.subplots(1,2,figsize=(20,5))

lns1 =ax1.plot(range(1,54),list(tesla_all_df["Twitter mentions"]),'--r', label = "Number of Twitter Mentions")
ax1.set_ylabel("Mentions")
ax2 = ax1.twinx()
lns2= ax2.plot(range(1,54),list(tesla_all_df["Absolute Weekly Difference"]),'g', label = "Abs Diff in Stock Price" )
ax2.set_ylabel("Absolute Weekly difference(USD)")
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg,labs,loc=0)
ax1.set_xlabel('Week')
ax1.set_title('Twitter Trends Against TSLA Stock Movement')

lns3 =ax3.plot(range(1,54),list(tesla_all_df["TSLA reddit mentions"]),'--b', label = "Number of Reddit Mentions")
ax3.set_ylabel("Mentions")
ax4 = ax3.twinx()
lns4= ax4.plot(range(1,54),list(tesla_all_df["Absolute Weekly Difference"]),'g', label = "Abs Diff in Stock Price" )
ax4.set_ylabel("Absolute  Weekly difference(USD)")
leg = lns3 + lns4
labs = [l.get_label() for l in leg]
ax3.legend(leg,labs,loc=0)
ax3.set_xlabel('Week')
ax3.set_title('Reddit Trends Against TSLA Stock Movement')

fig, (ax1,ax3) = plt.subplots(1,2,figsize=(20,5))

lns1 =ax1.plot(range(1,54),list(apple_all_df["Twitter mentions"]),'--r', label = "Number of Twitter Mentions")
ax1.set_ylabel("Mentions")
ax2 = ax1.twinx()
lns2= ax2.plot(range(1,54),list(apple_all_df["Absolute Weekly Difference"]),'g', label = "Abs Diff in Stock Price" )
ax2.set_ylabel("Absolute Weekly difference(USD)")
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg,labs,loc=0)
ax1.set_xlabel('Week')
ax1.set_title('Twitter Trends Against AAPL Stock Movement')
# ax1.set_ylim(-max(abs(apple_all_df["Twitter mentions"])),max(abs(apple_all_df["Twitter mentions"]))*1.2)
# ax2.set_ylim(-max(abs(apple_all_df["Absolute Weekly Difference"])),max(abs(apple_all_df["Absolute Weekly Difference"]))*1.2)

lns3 =ax3.plot(range(1,54),list(apple_all_df["AAPL reddit mentions"]),'--b', label = "Number of Reddit Mentions")
ax3.set_ylabel("Mentions")
ax4 = ax3.twinx()
lns4= ax4.plot(range(1,54),list(apple_all_df["Absolute Weekly Difference"]),'g', label = "Abs Diff in Stock Price" )
ax4.set_ylabel("Absolute Weekly difference(USD)")
leg = lns3 + lns4
labs = [l.get_label() for l in leg]
ax3.legend(leg,labs,loc=0)
ax3.set_xlabel('Week')
ax3.set_title('Reddit Trends Against AAPL Stock Movement')
# ax3.set_ylim(-max(abs(apple_all_df["AAPL reddit mentions"])),max(abs(apple_all_df["AAPL reddit mentions"]))*1.2)
# ax4.set_ylim(-max(abs(apple_all_df["Absolute Weekly Difference"])),max(abs(apple_all_df["Absolute Weekly Difference"]))*1.2)

fig, axes = plt.subplots(3,2,figsize=(15,17.5))
axes[0,0].set_title('Absolute Weekly Difference against TSLA reddit mentions')
sns.scatterplot(
    ax = axes[0,0], data=tesla_all_df, x="TSLA reddit mentions", y="Absolute Weekly Difference"
)
axes[1,0].set_title('Absolute Weekly Difference against AAPL reddit mentions')
sns.scatterplot(
    ax = axes[1,0], data=apple_all_df, x="AAPL reddit mentions", y="Absolute Weekly Difference"
)
axes[2,0].set_title('Absolute Weekly Difference against ZM reddit mentions')
sns.scatterplot(
    ax = axes[2,0], data=zoom_all_df, x="ZM reddit mentions", y="Absolute Weekly Difference"
)
axes[0,1].set_title('Absolute Weekly Difference against TSLA twitter mentions')
sns.scatterplot(
    ax = axes[0,1], data=tesla_all_df, x="Twitter mentions", y="Absolute Weekly Difference"
)
axes[1,1].set_title('Absolute Weekly Difference against AAPL twitter mentions')
sns.scatterplot(
    ax = axes[1,1], data=apple_all_df, x="Twitter mentions", y="Absolute Weekly Difference"
)
axes[2,1].set_title('Absolute Weekly Difference against ZM twitter mentions')
sns.scatterplot(
    ax = axes[2,1], data=zoom_all_df, x="Twitter mentions", y="Absolute Weekly Difference"
)

fig, axes = plt.subplots(3,2,figsize=(15,17.5))
axes[0,0].set_title('Absolute Weekly Difference against TSLA reddit mentions')
sns.scatterplot(
    ax = axes[0,0], data=tesla_all_df, x="TSLA reddit mentions", y="Absolute Weekly Difference"
)
axes[1,0].set_title('Absolute Weekly Difference against AAPL reddit mentions')
sns.scatterplot(
    ax = axes[1,0], data=apple_all_df, x="AAPL reddit mentions", y="Absolute Weekly Difference"
)
axes[2,0].set_title('Absolute Weekly Difference against ZM reddit mentions')
sns.scatterplot(
    ax = axes[2,0], data=zoom_all_df, x="ZM reddit mentions", y="Absolute Weekly Difference"
)
axes[0,1].set_title('Absolute Weekly Difference against TSLA twitter mentions')
sns.scatterplot(
    ax = axes[0,1], data=tesla_all_df, x="Twitter mentions", y="Absolute Weekly Difference"
)
axes[1,1].set_title('Absolute Weekly Difference against AAPL twitter mentions')
sns.scatterplot(
    ax = axes[1,1], data=apple_all_df, x="Twitter mentions", y="Absolute Weekly Difference"
)
axes[2,1].set_title('Absolute Weekly Difference against ZM twitter mentions')
sns.scatterplot(
    ax = axes[2,1], data=zoom_all_df, x="Twitter mentions", y="Absolute Weekly Difference"
)
for i in range(3):
    for j in range(2):
        axes[i,j].set_xscale('log')

from scipy.stats import spearmanr
corr_df = pd.DataFrame(columns = ['Reddit Correlation','Reddit P-Value','Twitter Correlation', 'Twitter P-Value'], index=['TSLA','AAPL','ZM'])

corr_df.loc['TSLA','Reddit Correlation'], corr_df.loc['TSLA','Reddit P-Value'] = spearmanr(tesla_all_df["TSLA reddit mentions"],tesla_all_df["Absolute Weekly Difference"])
corr_df.loc['AAPL','Reddit Correlation'], corr_df.loc['AAPL','Reddit P-Value'] = spearmanr(apple_all_df["AAPL reddit mentions"], apple_all_df["Absolute Weekly Difference"])
corr_df.loc['ZM','Reddit Correlation'], corr_df.loc['ZM','Reddit P-Value'] = spearmanr(zoom_all_df["ZM reddit mentions"],zoom_all_df["Absolute Weekly Difference"])
corr_df.loc['TSLA','Twitter Correlation'], corr_df.loc['TSLA','Twitter P-Value'] = spearmanr(tesla_all_df["Twitter mentions"],tesla_all_df["Absolute Weekly Difference"])
corr_df.loc['AAPL','Twitter Correlation'], corr_df.loc['AAPL','Twitter P-Value'] = spearmanr(apple_all_df["Twitter mentions"], apple_all_df["Absolute Weekly Difference"])
corr_df.loc['ZM','Twitter Correlation'], corr_df.loc['ZM','Twitter P-Value'] = spearmanr(zoom_all_df["Twitter mentions"],zoom_all_df["Absolute Weekly Difference"])
corr_df.head()

Data analysis

The main issues plaguing the analysis is heteroscedasticity, which violates the assumptions for ordinary least squares regression, and for Pearson's correlation. However, we can use Spearman's correlation, which only requires that:

There is ordinal data (this means the data has some sort of ranking, which twitter/reddit mentions and stock price differences has)
There is a monotonic relationship between the two variables of interest (which we can somewhat see from the scatterplot). while this cannot be used for prediction or regression, this allows us to investigate if there does exist a correlation between Twitter/Reddit mentions of the stock and the changes in the stock price itself.

Our study aimed to explore the connection between social media mentions on Reddit and Twitter and the stock prices of major companies such as Tesla, Apple, and Zoom. We analyzed data collected between January 1, 2020, and January 1, 2021, focusing on weekly trends to mitigate the impact of outliers in the volatile stock market.

Initially, we visualized trends in stock prices, Twitter mentions, and Reddit posts separately to understand their individual patterns. Subsequently, we merged these datasets to investigate the collective influence of social media activity on stock prices over the year.

Our analysis revealed a moderate positive correlation (ranging from 0.3 to 0.6) between social media mentions and stock prices for Tesla, Apple, and Zoom. This finding supported our hypothesis of a positive relationship between the frequency of Reddit and Twitter posts and average weekly closing prices of these stocks.

However, our study encountered limitations, notably the incomplete dataset and the inability to filter out irrelevant posts or consider retweets and likes on Twitter. Additionally, the lack of homoscedasticity in the data prevented the development of a predictive model for stock market investment.

Looking ahead, we aim to expand our dataset to include more comprehensive social media posts specifically mentioning the stocks by name, enabling more accurate predictive modeling. We also aspire to analyze additional social media platforms and stocks to provide investors with a reliable tool for decision-making.

While our study highlights the potential of social media in influencing stock markets, further research and refinement are necessary to develop robust predictive models tailored for investment strategies.
