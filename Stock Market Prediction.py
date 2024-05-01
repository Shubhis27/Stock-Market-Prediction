#!/usr/bin/env python
# coding: utf-8

# # DOWNLOADING S&P 500 PRICE DATA

# In[3]:


import yfinance as yf


# In[4]:


sp500 = yf.Ticker("^GSPC")


# In[6]:


sp500 = sp500.history(period="max")


# In[7]:


sp500


# In[8]:


sp500.index


# # CLEANING AND VISUALIZING OUR STOCK MARKET DATA

# In[9]:


sp500.plot.line(y="Close",use_index = True)


# In[11]:


del sp500["Dividends"]
del sp500["Stock Splits"]


# # SETTING UP OUR TARGET FOR MACHINE LEARNING

# In[12]:


sp500["Tomorrow"] = sp500["Close"].shift(-1)


# In[13]:


sp500


# In[15]:


sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500


# In[17]:


sp500 = sp500.loc["1999-12-31":].copy()
sp500


# # TRAINING AN INITIAL MACHINE LEARNING MODEL

# In[22]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train =sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])


# In[24]:


from sklearn.metrics import precision_score
preds = model.predict(test[predictors])


# In[26]:


import pandas as pd

preds = pd.Series(preds, index=test.index)


# In[27]:


precision_score(test["Target"], preds)


# In[28]:


combined = pd.concat([test["Target"], preds], axis=1)


# In[30]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, test.index, name = "Predictions")
    combined = pd.concat([test["Target"],preds],axis=1)
    return combined


# In[29]:


combined.plot()


# # BUILDING A BACKTESTING SYSTEM

# In[36]:


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train,test,predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)        


# In[37]:


predictions = backtest(sp500,model,predictors)


# In[38]:


predictions["Predictions"].value_counts()


# In[39]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[40]:


predictions["Target"].value_counts() / predictions.shape[0]


# # ADDING ADDITIONAL PREDICTORS TO OUR MODEL 

# In[42]:


horizons = [2,5,60,250,1000]

new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column , trend_column]


# In[48]:


sp500 = sp500.dropna()
sp500


# # Improving our Model

# In[44]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[45]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors]) [:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, test.index, name = "Predictions")
    combined = pd.concat([test["Target"],preds],axis=1)
    return combined


# In[ ]:


predictions = backtest(sp500, model, new_predictors)


# In[49]:


predictions["Predictions"].value_counts()


# In[50]:


precision_score(predictions["Target"], predictions["Predictions"])

