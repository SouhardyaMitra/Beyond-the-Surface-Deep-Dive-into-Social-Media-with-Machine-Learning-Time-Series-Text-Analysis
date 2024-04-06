#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import pandas as pd


# In[5]:


data= pd.read_csv("C:/Users/dell/Desktop/DS lab -2 project/new_dff.csv")


# In[4]:


data


# In[6]:


tweets_string = " ".join(data["Status.text"].tolist())
#Note that jupyter called Status text as Status.text


# In[7]:


#Stemming
# prompt: How to remove all urls, special characters and hashtags from tweets_string

import re

tweets_string = re.sub(r'http\S+', ' ', tweets_string)  # remove URLs
tweets_string = re.sub(r'[^A-Za-z0-9]+', ' ', tweets_string)  # remove special characters
tweets_string = re.sub(r'#', '', tweets_string)  # remove hashtags
tweets_string = re.sub(r'\d+', '',tweets_string)
unwanted_words = ["a", "and", "the", "of", "in", "to", "on", "for", "at", "with", "by", "from", "up", "down", "left", "right", "back", "forth", "over", "under", "around", "through", "out", "into", "onto", "off", "above", "below", "between", "among", "past", "around", "near", "far", "here", "there", "everywhere", "anywhere", "nowhere"]

# Remove unwanted words from the tweets_string
for word in unwanted_words:
  tweets_string = tweets_string.replace(" " + word + " ", " ")


# In[8]:



from collections import Counter
# Create a dictionary of word frequencies
word_counts = Counter(tweets_string.split())

# Create a DataFrame of word counts
df = pd.DataFrame.from_dict(word_counts, orient="index", columns=["Count"])

# Print the DataFrame
print(df)


# In[9]:


#Sentiment analysis
import pandas as pd
import numpy as np
import nltk
import re
import string
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)


# In[10]:


#describing the data
data.describe()


# In[11]:


#making subdata by taking status.text and sentiment
df = pd.DataFrame(data)

# Selecting specific columns
selected_columns = ['Status.text','Sentiment_Type']  # Columns you want to select
sub_data = df[selected_columns]

print(sub_data)


# In[12]:


sub_data.describe()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
#plotting the labels
plt.figure(figsize=(6,5))
sns.countplot(x='Sentiment_Type', data= sub_data)


# In[14]:


#renaming subdata as tweets
tweets= sub_data
#separating the text and the sentiment labels
text= tweets.iloc[:,0].values
sentiment= tweets.iloc[:,1].values


# In[15]:


#processing the data
text_processed = []
for i in range (0,len(text)):
    #removing the special characters
    text_extract = re.sub(r'\W',' ',str(text[i]))
    
    #removing single words
    text_extract = re.sub(r'\s+[a-zA-Z]\s+',' ', text_extract)
    
    #removing single characters from the beginning
    text_extract = re.sub(r'\^[a_zA-Z]\s+',' ', text_extract)
    
    #removing white space
    text_extract = re.sub(r'\s+',' ',text_extract, flags= re.I)
    
    #converting the text to lower case
    text_extract = text_extract.lower()
    
    text_processed.append(text_extract)


# In[16]:


text_processed


# In[17]:


#stemming and bag of words
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
stemming = nltk.stem.SnowballStemmer('english')

class Stemming(TfidfVectorizer):
    def build_analyzer(self):
        textanalyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda tweetdoc: (stemming.stem(i) for i in textanalyzer(tweetdoc))

textvectorizer = Stemming(max_features=10000, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
text_processed = textvectorizer.fit_transform(text_processed). toarray()

# Get the feature names from the fitted vectorizer
#Total number of words
words = textvectorizer.get_feature_names()
print("Total number of words:", len(words))


# In[18]:


# Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
NBX_train, NBX_test, NBY_train, NBY_test = train_test_split (text_processed, sentiment, test_size = 0.25, random_state=42)


# In[19]:


#Naive Bayes classifier
from sklearn.naive_bayes import BernoulliNB
NBC_sentiment = BernoulliNB()
NBC_sentiment.fit(NBX_train, NBY_train)


# In[20]:


#Predicting on test data
test_sentiment_predicted = NBC_sentiment.predict(NBX_test)


# In[28]:


#checking for accuracy
from sklearn import metrics
print(metrics.classification_report(NBY_test, test_sentiment_predicted))


# In[29]:


# The naive bayes model can predict neg at 45% , neutral at 88% and positive at 46% and a average of 60% of sentiments can be predicted


# In[24]:


#import matplotlib.pyplot as plt

# Assuming you have two datasets: 'text_data' and 'processed_data'
# Replace 'text_data' and 'processed_data' with your actual data
text_data = NBY_test  # List of original text data
processed_data = test_sentiment_predicted  # List of processed data from the Naive Bayes model

# Plot the actual text data vs processed data
plt.figure(figsize=(10, 6))
plt.plot(text_data, label='Actual Text Data', color='green')
plt.plot(processed_data, label='Processed Data', color='yellow')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.title('Actual Text Data vs Processed Data')
plt.legend()
plt.grid(False)
plt.show()


# In[27]:


import matplotlib.pyplot as plt

# Assuming you have two datasets: 'text_data' and 'processed_data'
# Replace 'text_data' and 'processed_data' with your actual data
# List of processed data from the Naive Bayes model

# Plot the actual text data vs processed data
plt.figure(figsize=(10, 6))

# Plot histogram for actual text data
plt.hist(text_data, bins=20, alpha=0.5, color='orange', label='Actual Text Data')

# Plot histogram for processed data
plt.hist(processed_data, bins=20, alpha=0.5, color='black', label='Processed Data')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Actual Text Data vs Processed Data')
plt.legend()
plt.grid(False)
plt.show()


# In[ ]:




