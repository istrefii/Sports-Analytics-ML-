#!/usr/bin/env python
# coding: utf-8

# ## Wearable Tech: Week 2 Assignment

# ### Step 1

# In[48]:


# imports
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

#turns off warning for chained assignment as we are overwriting references to original dataframe
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 30)

df = pd.read_csv("../Data/Soccer.csv", parse_dates=["Date"])
df.head()


# ### Step 2

# In[49]:


# use only sessions periods from the season
dfs= df[df['Period.Name']== 'Session']
dfs['Period.Name'].value_counts()


# ### Step 3

# In[50]:


# create session duration variable 
dfs['session_duration']= dfs['Total.Player.Load'] / dfs['Player.Load.Per.Minute']
dfs.head()


# In[52]:


# cleans up dataset
columns_to_keep =['Date', 'Player.Name', 'Period.Name', 'Position.Name', 'Total.Player.Load', 'session_duration']
dfs= dfs[columns_to_keep]
dfs


# ### Step 4

# In[56]:


# rank team players by cumulative playerload across the entire season
dfs.groupby(['Player.Name', 'Position.Name'])['Total.Player.Load'].sum().sort_values(ascending=False)


# ### Step 5

# In[17]:


# determine the 10 highest sessions for playerload and observe how many of these occured on game days
dfs.groupby('Date')[['Total.Player.Load', 'session_duration']].sum().sort_values(by='session_duration', ascending=False).head(10)


# ### Step 6

# In[58]:


# remove two players wiht lowest cumulative player load 
#(dfs.groupby(['Player.Name', 'Position.Name'])[['session_duration']].sum().sort_values(by='session_duration', ascending = False))
dfs.nlargest(20, 'Total.Player.Load')

# player 43 has 70% of the top 10 dates 


# ### Step 7 

# In[59]:


dfs= dfs.set_index('Player.Name')


# In[60]:


# drop certain players
dfs= dfs.drop([98,62,89,23,71])
dfs.head()


# In[63]:


# make date the index again 
dfs= dfs.set_index('Date', append=True)
dfs.head()


# In[64]:


# take player name out of index
dfs= dfs.reset_index('Player.Name')
dfs.head()


# In[65]:


# mkae player name as object
dfs['Player.Name']= dfs['Player.Name'].astype('object')


# In[67]:


# get data for player 43 only
df_43= dfs[dfs['Player.Name'] == 43]
df_43= df_43.resample('D').sum()
df_43.fillna(0, inplace=True)
df_43.head()


# In[68]:


# plot acute_pl and chronic_pl
df_43['Acute_PL'] = (df_43.loc[:, "Total.Player.Load"].rolling('3D',3).sum())/3
df_43['Chronic_PL'] = (df_43.loc[:, 'Total.Player.Load'].rolling('21D',21).sum())/3
df_43['Acute:Chronic']= df_43['Acute_PL']/ df_43.Chronic_PL
df_43.head()


# ### Step 8

# In[69]:


fig, axs = plt.subplots(2,figsize=(20, 10))

axs[0].bar(df_43.index, df_43["Total.Player.Load"])
#axs[0].set_xlabel("Date", fontsize=12)
axs[0].tick_params(axis='x', labelsize=12)
axs[0].set_ylabel('Average Player Load', fontsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[0].set_title("Daily Average Player Load, Player 12 (D)", fontsize=12)

color='#FA8775'
axs[1].set_xlabel("Date", fontsize=12)
axs[1].tick_params(axis='x', labelsize=12, color=color)
axs[1].set_ylabel('Acute PL', color=color, fontsize=12)
axs[1].bar(df_43.index, df_43["Acute_PL"], color=color, alpha=0.6)
axs[1].tick_params(axis='y', labelsize=12)
color='#FFB14E'
axs[1].fill_between(df_43.index, df_43["Chronic_PL"], color=color, alpha=0.8)
axs[1] = plt.gca()
axs[1].set_title("Acute to Chronic Ratio, Player 12 (D)", fontsize=12)
axs1 = axs[1].twinx()  # instantiate a second axes that shares the same x-axis

color='#882C78'
axs1.set_ylabel('Ratio', fontsize=12, color=color)  # we already handled the x-label with ax1
axs1.plot(df_43.index, df_43["Acute:Chronic"], color=color, linewidth=3.0)
axs1.tick_params(axis='y', labelsize=12)
axs1 = plt.gca()

# To see how the acute to chronic workload ratio drops over the first 28 days, 
# comment out this line by adding a '#' in front of the line
axs1.set_ylim([0,3.0])

axs1.axhline(y=1.5,c="gray",linewidth=2,zorder=0)

# pink bar are Acute_PL
# orange shadding is Chronic_PL
# purple line is Acute:Chronic


# In[73]:


df_all = dfs.groupby("Date").sum()
df_all = df_all.resample('D').sum()
df_all = df_all.fillna(0)

df_all["Acute_PL"] = (df_all.loc[:,"Total.Player.Load"].rolling('3D', 3).sum())/3
df_all["Chronic_PL"] = (df_all.loc[:,"Total.Player.Load"].rolling('21D', 21).sum())/21
df_all["Acute:Chronic"] = df_all.loc[:,"Acute_PL"]/df_all.loc[:,"Chronic_PL"]
df_all.head()


# In[74]:


fig, axs = plt.subplots(2,figsize=(20, 10))

axs[0].bar(df_all.index, df_all["Total.Player.Load"])
#axs[0].set_xlabel("Date", fontsize=12)
axs[0].tick_params(axis='x', labelsize=12)
axs[0].set_ylabel('Total Player Load', fontsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[0].set_title("Total Player Load, Team", fontsize=12)

color='#FA8775'
axs[1].set_xlabel("Date", fontsize=12)
axs[1].tick_params(axis='x', labelsize=12, color=color)
axs[1].set_ylabel('Acute PL', color=color, fontsize=12)
axs[1].bar(df_all.index, df_all["Acute_PL"], color=color, alpha=0.6)
axs[1].tick_params(axis='y', labelsize=12)

color='#FFB14E'
axs[1].fill_between(df_all.index, df_all["Chronic_PL"], color=color, alpha=0.8)
axs[1] = plt.gca()
axs[1].set_title("Acute to Chronic Ratio, Team", fontsize=12)
axs1 = axs[1].twinx()  # instantiate a second axes that shares the same x-axis

color='#882C78'
axs1.set_ylabel('Ratio', fontsize=12, color=color)  # we already handled the x-label with ax1
axs1.plot(df_all.index, df_all["Acute:Chronic"], color=color, linewidth=3.0)
axs1.tick_params(axis='y', labelsize=12)
axs1 = plt.gca()

# To see how the acute to chronic workload ratio drops over the first 28 days, 
# comment out this line by adding a '#' in front of the line
axs1.set_ylim([0,2.0])

axs1.axhline(y=1.5,c="gray",linewidth=2,zorder=0)


# ### Step 9

# In[75]:


df_all.nlargest(5, 'Acute:Chronic')


# ### Step 10

# In[77]:


df_all.nsmallest(50, 'session_duration')
# determine how many days of rest the team had (days with zero) Answer = 26,
# determine what day the team had the lowest total session duration: Aug 21st. (day before first real game)


# In[ ]:




