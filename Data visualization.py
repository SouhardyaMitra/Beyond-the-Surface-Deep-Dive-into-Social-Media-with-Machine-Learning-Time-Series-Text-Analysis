#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data= pd.read_csv("C:/Users/dell/Desktop/DS lab -2 project/new_dff.csv")


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


#'State', 'retweet_count', 'reply_count','like_count', 'quote_count', 'Buzz', 'Day', 'Time', 'Followers','Vividness', 'WC', 'Clout', 'Cognition', 'emotion', 'emo_pos','emo_neg', 'Positive', 'Negative','Total_sentiment1', 'Sentiment_Type'


# In[19]:


#Making two new datasets containing numerical and textual values
columns1 = ['retweet_count', 'reply_count', 'retweet_count', 'reply_count','like_count', 'quote_count', 'Buzz', 'Followers', 'WC', 'Clout', 'Cognition', 'emotion', 'emo_pos','emo_neg', 'Positive', 'Negative','Total_sentiment1' ] 

data1 = data.loc[:, columns1]

columns2 = ['State','Time', 'Day', 'Vividness','Sentiment_Type' ]

data2 = data.loc[:, columns2]


# In[20]:


missing_values = data1.isnull().sum()
print(missing_values)


# In[21]:


missing_values = data2.isnull().sum()
print(missing_values)


# In[23]:


print(len(data1.index))
print(len(data1[column]))


# In[35]:


#visualization of retweet count
# frequency bar plot
import pandas as pd
import matplotlib.pyplot as plt


# Specify the column you want to plot
column_to_plot = 'retweet_count'

# Count the frequency of each unique value in the specified column
value_counts = data[column_to_plot].value_counts()

# Sort the values by index (optional, for aesthetics)
value_counts.sort_index(inplace=True)

# Plot the frequency bar plot
plt.bar(value_counts.index, value_counts.values, color='grey')

# Add labels and title
plt.xlabel(column_to_plot)
plt.ylabel('Frequency')
plt.title('Frequency Bar Plot of {}'.format(column_to_plot))
plt.xlim(left=0, right=len(value_counts)-1)

# Show the plot
plt.tight_layout()
plt.show()


# In[36]:


data[column_to_plot].describe()


# In[37]:


# Specify the column you want to plot
column_to_plot = 'reply_count'

# Count the frequency of each unique value in the specified column
value_counts = data[column_to_plot].value_counts()

# Sort the values by index (optional, for aesthetics)
value_counts.sort_index(inplace=True)

# Plot the frequency bar plot
plt.bar(value_counts.index, value_counts.values, color='grey')

# Add labels and title
plt.xlabel(column_to_plot)
plt.ylabel('Frequency')
plt.title('Frequency Bar Plot of {}'.format(column_to_plot))
plt.xlim(left=0, right=len(value_counts)-1)

# Show the plot
plt.tight_layout()
plt.show()


# In[38]:


data[column_to_plot].describe()


# In[55]:


# Specify the column you want to plot
column_to_plot = 'like_count'

# Count the frequency of each unique value in the specified column
value_counts = data[column_to_plot].value_counts()

# Sort the values by index (optional, for aesthetics)
value_counts.sort_index(inplace=True)

# Plot the frequency bar plot
plt.bar(value_counts.index, value_counts.values, color='grey')

# Add labels and title
plt.xlabel(column_to_plot)
plt.ylabel('Frequency')
plt.title('Frequency Bar Plot of {}'.format(column_to_plot))
plt.xlim(left=0, right=len(value_counts)-1)

# Show the plot
plt.tight_layout()
plt.show()


# In[40]:


data[column_to_plot].describe()


# In[41]:


# Specify the column you want to plot
column_to_plot = 'Buzz'

# Count the frequency of each unique value in the specified column
value_counts = data[column_to_plot].value_counts()

# Sort the values by index (optional, for aesthetics)
value_counts.sort_index(inplace=True)

# Plot the frequency bar plot
plt.bar(value_counts.index, value_counts.values, color='grey')

# Add labels and title
plt.xlabel(column_to_plot)
plt.ylabel('Frequency')
plt.title('Frequency Bar Plot of {}'.format(column_to_plot))
plt.xlim(left=0, right=len(value_counts)-1)

# Show the plot
plt.tight_layout()
plt.show()


# In[42]:


data[column_to_plot].describe()


# In[44]:


#followers statewise 
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame with columns 'state', 'followers', and 'word_count'
# Replace 'tweets.csv' with the path to your dataset
df = data

# Group by 'state' and calculate the total number of followers for each state
followers_by_state = df.groupby('State')['Followers'].sum().reset_index()

# Plot the state versus the total number of followers
plt.figure(figsize=(10, 6))
plt.bar(followers_by_state['State'], followers_by_state['Followers'], color='red')
plt.xlabel('State')
plt.ylabel('Number of Followers')
plt.title('Total Number of Followers by State')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[46]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame with columns 'state', 'followers', and 'word_count'
# Replace 'tweets.csv' with the path to your dataset
df = data

# Plot the state versus the number of followers
plt.figure(figsize=(10, 6))
plt.bar(df['State'], df['Followers'], color='navy')
plt.xlabel('State')
plt.ylabel('Number of Followers')
plt.title('Number of Followers by State')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[ ]:




