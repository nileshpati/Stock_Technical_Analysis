# Stock_Technical_Analysis
Technical Analysis for Stocks
# Value (In Lacs) = Volume
# symbol =  Symbol
# https://towardsdatascience.com/building-a-comprehensive-set-of-technical-indicators-in-python-for-quantitative-trading-8d98751b5fb
# Simple Moving Average\
##https://github.com/FreddieWitherden/ta/blob/master/ta.py
def SMA(all_data,period1,period2):
    all_data['SMA_P1'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = period1).mean())
    all_data['SMA_P2'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = period2).mean())
    all_data['SMA_ratio'] = all_data['SMA_P1'] / all_data['SMA_P2']
    return (all_data)
# Simple Moving Average Volume
def SMA_Volume(all_data,period1,period2):
    all_data['SMAP1_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = period1).mean())
    all_data['SMAP2_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = period2).mean())
    all_data['SMA_Volume_Ratio'] = all_data['SMAP1_Volume']/all_data['SMAP2_Volume']
    return (all_data)    
# Wilder’s Smoothing on MA    
def Wilder(data_column, period):
    #start = np.where(~np.isnan(data))[0][0] #Check if nans present in beginning
    #start = 0 
    # Wilder = np.array([np.nan]*len(data))
    # Wilder[start+periods-1] = data[start:(start+periods)].mean() #Simple Moving Average
    # for i in range(start+periods,len(data)):
    #     Wilder[i] = (Wilder[i-1]*(periods-1) + data[i])/periods #Wilder Smoothing
    # return(Wilder)  
    return(pd.Series(data=[data_column.iloc[:period].mean()],
          index=[data_column.index[period-1]],
    ).append(data_column.iloc[period:]
          ).ewm(
    alpha=1.0 / period,
    adjust=False,
    ).mean())


#Average True Range (ATR) : to measure volatility in the market
#ATR however is primarily used in identifying when to exit or enter a trade rather than the direction in which to trade the stock
def ATR(all_data,period1, period2):
    all_data['prev_close'] = all_data.groupby('symbol')['Close'].shift(1)
    all_data['TR'] = np.maximum((all_data['High'] - all_data['Low']), 
                     np.maximum(abs(all_data['High'] - all_data['prev_close']), 
                     abs(all_data['prev_close'] - all_data['Low'])))
    for i in all_data['symbol'].unique():
        TR_data = all_data[all_data.symbol == i].copy()
        all_data.loc[all_data.symbol==i,'ATR_P1'] = Wilder(TR_data['TR'], period1)
        all_data.loc[all_data.symbol==i,'ATR_P2'] = Wilder(TR_data['TR'], period2)

    all_data['ATR_Ratio'] = all_data['ATR_P1'] / all_data['ATR_P2']
    return(all_data)   
# Average Directional Index (ADX) In general, an ADX of 25 or above indicates a strong trend and an ADX of less than 20 indicates a weak trend. 
# The calculation of ADX is quite complex and requires certain steps.
def ADX(all_data,period1,period2):   
        all_data['prev_high'] = all_data.groupby('symbol')['High'].shift(1)
        all_data['prev_low'] = all_data.groupby('symbol')['Low'].shift(1)

        all_data['+DM'] = np.where(~np.isnan(all_data.prev_high),
                                   np.where((all_data['High'] > all_data['prev_high']) & 
                 (((all_data['High'] - all_data['prev_high']) > (all_data['prev_low'] - all_data['Low']))), 
                                                                          all_data['High'] - all_data['prev_high'], 
                                                                          0),np.nan)

        all_data['-DM'] = np.where(~np.isnan(all_data.prev_low),
                                   np.where((all_data['prev_low'] > all_data['Low']) & 
                 (((all_data['prev_low'] - all_data['Low']) > (all_data['High'] - all_data['prev_high']))), 
                                            all_data['prev_low'] - all_data['Low'], 
                                            0),np.nan)

        for i in all_data['symbol'].unique():
            ADX_data = all_data[all_data.symbol == i].copy()
            all_data.loc[all_data.symbol==i,'+DM_P1'] = Wilder(ADX_data['+DM'], period1)
            all_data.loc[all_data.symbol==i,'-DM_P1'] = Wilder(ADX_data['-DM'], period1)
            all_data.loc[all_data.symbol==i,'+DM_P2'] = Wilder(ADX_data['+DM'], period2)
            all_data.loc[all_data.symbol==i,'-DM_P2'] = Wilder(ADX_data['-DM'], period2)

        all_data['+DI_P1'] = (all_data['+DM_P1']/all_data['ATR_P1'])*100
        all_data['-DI_P1'] = (all_data['-DM_P1']/all_data['ATR_P1'])*100
        all_data['+DI_P2'] = (all_data['+DM_P2']/all_data['ATR_P2'])*100
        all_data['-DI_P2'] = (all_data['-DM_P2']/all_data['ATR_P2'])*100

        all_data['DX_P1'] = (np.round(abs(all_data['+DI_P1'] - all_data['-DI_P1'])/(all_data['+DI_P1'] + all_data['-DI_P1']) * 100))

        all_data['DX_P2'] = (np.round(abs(all_data['+DI_P2'] - all_data['-DI_P2'])/(all_data['+DI_P2'] + all_data['-DI_P2']) * 100))
        for i in all_data['symbol'].unique():
            ADX_data = all_data[all_data.symbol == i].copy()
            all_data.loc[all_data.symbol==i,'ADX_P1'] = Wilder(ADX_data['DX_P1'], period1)
            all_data.loc[all_data.symbol==i,'ADX_P2'] = Wilder(ADX_data['DX_P2'], period2)
        return(all_data)     

#Stochastic oscillator is a momentum indicator aiming at identifying overbought and oversold securities and is commonly used in technical analysis.
def Stochastic_Oscillators(all_data,period1,period2):
  all_data['Lowest_P1'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = period1).min())
  all_data['High_P1'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = period1).max())
  all_data['Lowest_P2'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = period2).min())
  all_data['High_P2'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = period2).max())

  all_data['Stochastic_P1'] = ((all_data['Close'] - all_data['Lowest_P1'])/(all_data['High_P1'] - all_data['Lowest_P1']))*100
  all_data['Stochastic_P2'] = ((all_data['Close'] - all_data['Lowest_P2'])/(all_data['High_P2'] - all_data['Lowest_P2']))*100

  all_data['Stochastic_%D_P1'] = all_data['Stochastic_P1'].rolling(window = period1).mean()
  all_data['Stochastic_%D_P2'] = all_data['Stochastic_P2'].rolling(window = period2).mean()

  all_data['Stochastic_Ratio'] = all_data['Stochastic_%D_P1']/all_data['Stochastic_%D_P2']
  return(all_data)
#Most common used momentum indicator  
def RSI(all_data,period1, period2):
    all_data['Diff'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.diff())
    all_data['Up'] = all_data['Diff']
    all_data.loc[(all_data['Up']<0), 'Up'] = 0

    all_data['Down'] = all_data['Diff']
    all_data.loc[(all_data['Down']>0), 'Down'] = 0 
    all_data['Down'] = abs(all_data['Down'])

    all_data['avg_P1up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=period1).mean())
    all_data['avg_P1down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=period1).mean())

    all_data['avg_P2up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=period2).mean())
    all_data['avg_P2down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=period2).mean())

    all_data['RS_P1'] = all_data['avg_P1up'] / all_data['avg_P1down']
    all_data['RS_P2'] = all_data['avg_P2up'] / all_data['avg_P2down']

    all_data['RSI_P1'] = 100 - (100/(1+all_data['RS_P1']))
    all_data['RSI_P2'] = 100 - (100/(1+all_data['RS_P2']))

    all_data['RSI_ratio'] = all_data['RSI_P1']/all_data['RSI_P2']
    return(all_data)  
#MACD uses two exponentially moving averages and creates a trend analysis based on their convergence or divergence. Although most commonly used MACD slow and fast signals are based on 26 days and 12 days respectively, I have used 15 days and 5 days to be consistent with other indicators.
def MACD(all_data,period1,period2): 
  all_data['P1_Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=period1, adjust=False).mean())
  all_data['P2_Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=period2, adjust=False).mean())
  all_data['MACD'] = all_data['P2_Ewm'] - all_data['P1_Ewm']
  return(all_data)
#Bollinger Bands Bollinger bands capture the volatility of a stock and are used to identify overbought and oversold stocks. Bollinger bands consists of three main elements: The simple moving average line, an upper bound which is 2 standard deviations above moving average and a lower bound which is 2 standard deviations below moving average.  
def Bollinger_Band(all_data,period1,period2):
  all_data['P2MA'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=period2).mean())
  all_data['P2SD'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=period2).std())
  all_data['upperband'] = all_data['P2MA'] + 2*all_data['P2SD']
  all_data['lowerband'] = all_data['P2MA'] - 2*all_data['P2SD']
  return(all_data)
# Rate_of_change Rate of change is a momentum indicator that explains a price momentum relative to a price fixed period before.
def Rate_of_change(all_data,period1,period2):
  all_data['RC_P1'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods = period1))
  all_data['RC_P2'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods = period2))
  return(all_data) 
# Using Pandas to calculate EMA. adjust=False specifies that we are interested in the recursive calculation mode.
def ema(all_data, period1,period2):
  all_data['EMA_P1'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=period1, adjust=False).mean())
  all_data['EMA_P2'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=period2, adjust=False).mean())
  
  return(all_data)
# VWP
def vwap(all_data):
  v = all_data['Volume'].values
  tp = (all_data['Low'] + all_data['Close'] + all_data['High']).div(3).values
  return all_data.assign(vwap=(tp * v).cumsum() / v.cumsum())
##CCI
def CCI(data, period1):
  TP = (data['High'] + data['Low'] + data['Close']) / 3
  CCI = pd.Series((TP - TP.rolling(window=period1, center = False).mean()) / (0.015 * TP.rolling(window=period1, center=False).std()), name = 'CCI')
  data = data.join(CCI)
  return data
## Arron 


##OBV
def add_obv(df):
    copy = df.copy()
    # https://stackoverflow.com/a/66827219
    copy["OBV"] = (np.sign(copy["Close"].diff()) * copy["Volume"]).fillna(0).cumsum()
    return copy
    
def all_tech_variable(all_data, period1,period2):
    all_data = SMA(all_data, period1,period2)
    all_data = SMA_Volume(all_data, period1,period2)
    all_data = ATR(all_data, period1,period2)
    all_data = ADX(all_data, period1,period2)
    all_data = Stochastic_Oscillators(all_data, period1,period2)
    all_data = RSI(all_data, period1,period2)
    all_data = MACD(all_data, period1,period2)
    all_data = Bollinger_Band(all_data, period1,period2)
    all_data = Rate_of_change(all_data, period1,period2)
    return(all_data)
    
    
#importing packages
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as pdr
import seaborn as sns
import matplotlib.pyplot as plt
import bs4 as bs
import requests
from IPython.display import clear_output
from scipy.stats import mstats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RandomizedSearchCV, validation_curve, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from sklearn.model_selection import GridSearchCV
import yahoo_fin
import datetime as dt
import yfinance as yf
from pandas_datareader import data as pdr
sns.set()

#Obtain list of S&100 companies from wikipedia
#resp = requests.get("https://www1.nseindia.com/content/indices/ind_nifty500list.csv")

# convert_soup = bs.BeautifulSoup(resp.text, 'lxml')
# table = convert_soup.find('table',{'class':'wikitable sortable'})

# tickers = []

# for rows in table.findAll('tr')[1:]:
#     ticker = rows.findAll('td')[0].text.strip()
#     tickers.append(ticker)

# all_data = pd.DataFrame()
# test_data = pd.DataFrame()

#no_data = []
#all_data = pd.DataFrame()

url = "https://www1.nseindia.com/content/indices/ind_nifty500list.csv"
df1 = pd.read_csv(url)
#tickers = list(df1.Symbol.sort_values())
#no_data = list(no_data.unique())
#Extract data from Yahoo Finance
for i in tickers :
    try:
        ##time.sleep(1)
        ##test_data = pdr.get_data_yahoo(i, start = dt.datetime(1990,1,1), end = dt.date.today())
        test_data = yf.download(str(i) + ".NS" , start = dt.datetime(1990,1,1), end = dt.date.today())
        ##time.sleep(2) 
        test_data['symbol'] = i
        all_data = all_data.append(test_data)
        clear_output(wait = True)
    except:
        no_data.append(i)
        ##print(len(no_data))

    clear_output(wait = True)
