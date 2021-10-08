import requests
import pandas as pd
import json
from datetime import date, datetime, tzinfo
import csv
from zoneinfo import ZoneInfo

from requests.models import Response

class StockNewsAPI:
    def __init__(self, api_key, remove_site_url=False, remove_image_url=True, remove_topics=False):
        
        self.api_key = api_key

        self.remove_site_url = remove_site_url
        self.remove_topics = remove_topics
        self.remove_image_url = remove_image_url


        self.tickers = None
        

    def _get_api_call(self, url, tickers):
        if self.tickers == None:
            raise ValueError('no tickers in StockNewsAPI instance')            

        if not isinstance(tickers, str):
            tickers_string = '_'.join(self.tickers)
        else:
            tickers_string = tickers

        
        now = datetime.now()
        date_time = now.strftime('%m_%d_%Y_%H_%M_%S')
        
        file_name = f'{tickers_string}_{date_time}.csv'                
            
        df = pd.read_csv(url)
        df.to_csv(file_name, index=False)

        return file_name


    def get_company_ticker_news(self, tickers, api_call_per_stock=False, item_count=50):
        if item_count > 50:
            raise ValueError("Max news articles per call is 50")

        if type(tickers) is list: 
            self.tickers = [t.upper() for t in tickers]
            tickers_string = ','.join(self.tickers)
        else:
            self.tickers = [tickers.upper()]
            tickers_string = tickers.upper()

        if not api_call_per_stock:
            url = f"https://stocknewsapi.com/api/v1?tickers={tickers_string}&items={item_count}&token={self.api_key}&datatype=csv&extra-fields=rankscore"
            file_name = self._get_api_call(url, self.tickers)
            return self.load_existing_file(file_name)


        file_names = [self._get_api_call(f"https://stocknewsapi.com/api/v1?tickers={t}&items={item_count}&token={self.api_key}&datatype=csv&extra-fields=rankscore", t)
                        for t in self.tickers]

        return self._load_existing_files(file_names)

        

    def load_existing_file(self, file_name):
        df = pd.read_csv(file_name)

        if self.remove_site_url:
            df = df.drop(columns=['news_url'])

        if self.remove_image_url:
            df = df.drop(columns=['image_url'])

        if self.remove_topics:
            df = df.drop(columns=['topics'])

        now = datetime.utcnow()
        today = now.strftime('%m_%d_%y')
        today = datetime.strptime(today, '%m_%d_%y')

        df.tickers = df.tickers.str.split(',')
        df = df.explode('tickers', ignore_index=True)


        df.date = pd.to_datetime(df.date, infer_datetime_format=True)
        df.date = df.date.dt.strftime('%m_%d_%y')
        df.date = pd.to_datetime(df.date, format='%m_%d_%y')

        df.sentiment = df.sentiment.map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
        df['rank_weighted_sentiment'] = df.sentiment * df.rank_score
        

        df['days_since'] = (today - df.date).dt.days
        df = df.set_index('date')

        df_dict = {}
        for ticker in df.tickers.unique():
            data_df = df[df.tickers == ticker].copy()
            data_df.sort_index(ascending=True, inplace=True)
            data_df['cumulative_sentiment'] = data_df.sentiment.cumsum()
            data_df['cumulative_rank_weighted_sentiment'] = data_df.rank_weighted_sentiment.cumsum()
            df_dict[ticker] = data_df


        return df_dict


    def _load_existing_files(self, file_names):
        final_df_dict = {}
        for ticker,file_name in zip(self.tickers,file_names):
            df_dict = self.load_existing_file(file_name)
            if df_dict is not None and ticker in df_dict:
                final_df_dict[ticker] = df_dict[ticker].copy()

        return final_df_dict



