#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.tsa.arima.model import ARIMA
from textstat import flesch_reading_ease

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

get_ipython().system('pip install textstat')


# In[2]:


data = pd.read_csv(r"C:\Users\HP\Desktop\projectd\Data_Project1.csv")


# In[3]:


data['Date1'] = pd.to_datetime(data['Date1'])
data.set_index('Date1', inplace=True)


# In[4]:


# Step 2: Text Analysis
text_data = data['Status text']

# Tokenization
tokens = [word_tokenize(text) for text in text_data]

# Lowercasing
tokens_lower = [[token.lower() for token in text] for text in tokens]

# Removing stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [[token for token in text if token not in stop_words] for text in tokens_lower]


# In[5]:


token_strings = [' '.join(tokens) for tokens in filtered_tokens]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(token_strings)
tfidf_scores = tfidf_matrix.toarray()


# In[6]:


sia = SentimentIntensityAnalyzer()
sentiment_scores = [sia.polarity_scores(text)['compound'] for text in text_data]


# In[7]:


word_frequency = [len(tokens) for tokens in filtered_tokens]
lexical_diversity = [len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0 for tokens in filtered_tokens]
readability_scores = [flesch_reading_ease(' '.join(tokens)) for tokens in filtered_tokens]


# In[8]:


data['TF-IDF Score'] = np.mean(tfidf_scores, axis=1)
data['Sentiment Score'] = sentiment_scores
data['Word Frequency'] = word_frequency
data['Lexical Diversity'] = lexical_diversity
data['Readability Score'] = readability_scores
numeric_columns = data.select_dtypes(include=[np.number]).columns  # Select numeric columns
daily_data = data[numeric_columns].resample('D').mean()  # Aggregate by day


# In[9]:


import pandas as pd

# Assuming 'data' is your DataFrame
# Drop rows with any missing values
daily_data.dropna(inplace=True)


# In[10]:


daily_data


# In[11]:


columns_to_keep = {"TF-IDF Score",	"Sentiment Score",	"Word Frequency",	"Lexical Diversity",	"Readability Score"}
newdata = daily_data.drop(columns=daily_data.columns.difference(columns_to_keep))


# In[12]:


newdata


# In[ ]:





# In[13]:


import matplotlib.pyplot as plt

# Plotting time series for each metric
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))

metrics = ['TF-IDF Score', 'Sentiment Score', 'Word Frequency', 'Lexical Diversity', 'Readability Score']
colors = ['blue', 'green', 'red', 'purple', 'orange']

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.plot(newdata.index, newdata[metric], color=colors[i])
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.set_title(f'Time Series: {metric}')

plt.tight_layout()
plt.show()


# In[14]:


import seaborn as sns

# Compute correlation matrix
correlation_matrix = newdata[['TF-IDF Score', 'Sentiment Score', 'Word Frequency', 'Lexical Diversity', 'Readability Score']].corr()

# Plotting correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('5.png')
plt.show()


# In[35]:


get_ipython().system('pip install scikit-learn')

from sklearn.ensemble import IsolationForest

# Assuming 'df' is your DataFrame with DateTimeIndex and metrics columns
# Combine all metrics into a single DataFrame
metrics_df = newdata[['TF-IDF Score', 'Sentiment Score', 'Word Frequency', 'Lexical Diversity', 'Readability Score']]

# Initialize Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05)  # Adjust contamination parameter as needed

# Fit the model and predict anomalies
isolation_forest.fit(metrics_df)
anomaly_labels = isolation_forest.predict(metrics_df)
anomalies = metrics_df[anomaly_labels == -1]
anomaly_scores = isolation_forest.decision_function(metrics_df)
threshold=0.12
anomalies_below_threshold = metrics_df[anomaly_scores < threshold]
print("Anomaly Scores:")
print(anomaly_scores)
print("Anomalies below threshold of 0.12:")
print(anomalies_below_threshold)


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming your time series data is stored in a DataFrame named 'newdata' with a DateTimeIndex
# Let's focus on one metric, for example, 'TF-IDF Score'

# Calculate the rolling mean (moving average) over a specific window size
window_size = 30  # Adjust this window size based on your data frequency
newdata['Sentiment Score MA'] = newdata['Sentiment Score'].rolling(window=window_size).mean()

# Plotting the original TF-IDF Score and its moving average
plt.figure(figsize=(10, 6))
plt.plot(newdata.index, newdata['Sentiment Score'], label='Original TF-IDF Score', color='blue')
plt.plot(newdata.index, newdata['Sentiment Score MA'], label=f'Moving Average ({window_size} days)', color='orange')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Trend Analysis: Sentiment Score with Moving Average')
plt.legend()
plt.grid(True)
plt.savefig('7.png')
plt.show()


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Assuming your time series data is stored in a DataFrame named 'newdata' with a DateTimeIndex
# Handle missing values in the DataFrame by interpolating
newdata_interpolated = newdata.interpolate(method='linear')

# Perform seasonal decomposition using multiplicative model on the interpolated data
result = seasonal_decompose(newdata_interpolated['Sentiment Score'], model='multiplicative')

# Plotting the seasonal component
plt.figure(figsize=(10, 6))
plt.plot(result.seasonal.index, result.seasonal.values, label='Seasonal Component', color='orange')
plt.xlabel('Date')
plt.ylabel('Seasonal Component')
plt.title('Seasonal Pattern Detection: TF-IDF Score')
plt.legend()
plt.grid(True)
plt.show()


# In[35]:


pip install prophet


# In[60]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


# Assuming you have a DataFrame named 'newdata' with a DateTimeIndex and a metric 'Value'
# Replace 'Value' with your actual metric name
if 'level_0' in newdata.columns:
    newdata.drop(columns=['level_0'], inplace=True)

# Reset the index of newdata if DateTimeIndex is not in 'ds' format
newdata.reset_index(inplace=True)
newdata.rename(columns={'Date1': 'ds', 'Sentiment Score': 'y'}, inplace=True)

# Initialize and fit Prophet model
model_prophet = Prophet()
model_prophet.fit(newdata)

# Make future predictions
future = model_prophet.make_future_dataframe(periods=30)  # Example periods for forecasting
forecast_prophet = model_prophet.predict(future)

# Print the forecasted values
print(forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# In[61]:


plt.figure(figsize=(10, 6))
plt.plot(newdata['ds'], newdata['y'], label='Historical Data')
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Forecast', linestyle='--')
plt.fill_between(forecast_prophet['ds'], forecast_prophet['yhat_lower'], forecast_prophet['yhat_upper'], color='gray', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Historical Data vs Forecast with Prophet')
plt.legend()
plt.grid(True)
plt.show()


# In[58]:


from sklearn.metrics import mean_absolute_error

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(newdata['y'], forecast_prophet['yhat'][:len(newdata)])

print(f"Mean Absolute Error (MAE): {mae:.2f}")



# In[65]:


pip install tensorflow


# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

if 'level_0' in newdata.columns:
    newdata.drop(columns=['level_0'], inplace=True)

# Assuming you have a DataFrame named 'newdata' with a DateTimeIndex and a metric 'Value'
# Replace 'Value' with your actual metric name

# Reset the index of newdata if DateTimeIndex is not in 'ds' format
newdata.reset_index(inplace=True)
newdata.rename(columns={'Date1': 'ds', 'Sentiment Score': 'y'}, inplace=True)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(newdata['y'].values.reshape(-1, 1))

# Define the sequence length for input data
sequence_length = 10  # You can adjust this based on your data and requirements

# Create sequences of input data and corresponding labels
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and testing sets (e.g., 80% train, 20% test)
split_index = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Define the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions with the trained model
predictions = model.predict(X_test)

# Inverse transform the predictions to original scale
predictions = scaler.inverse_transform(predictions)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(newdata.index[split_index+sequence_length:], y_test, label='Actual')
plt.plot(newdata.index[split_index+sequence_length:], predictions, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values (LSTM Forecast)')
plt.legend()
plt.grid(True)
plt.show()

# Print the predicted values
print(predictions)


# In[70]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

