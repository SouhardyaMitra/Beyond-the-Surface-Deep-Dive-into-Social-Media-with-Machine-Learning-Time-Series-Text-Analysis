#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


df=pd.read_csv(r"C:\Users\USER\Downloads\new_dff_total.csv")


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


df.columns


# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score



# In[15]:


import random


# In[16]:


df.describe().transpose().style.background_gradient(cmap='tab20c')    


# In[70]:


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(24, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[72]:


correlation_matrix


# # Machine Learning Algorithms

# In[17]:


df.info()


# In[18]:


# Engagement ratio feature
df['engagement_ratio'] = (df['retweet_count'] + df['reply_count'] + df['like_count']) / df['Followers']
df['engagement_ratio'].head()


# In[19]:


df.columns


# In[20]:


categorical_columns = df[['Vividness','Day','State']]
categorical_columns


# In[21]:


columns_to_drop = ['Vividness','Day','State','Status.text','Time']

numerical_column=df.drop(columns=columns_to_drop)
numerical_column


# ## PCA(principal component analysis)

# In[22]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[23]:


numerical_columns = ['retweet_count','reply_count','like_count','quote_count','Buzz','Followers','WC','Clout','Cognition','emotion','emo_pos','emo_neg','Positive','Negative']
categorical_columns = ['Vividness','Day','State']


# In[24]:


numerical_columns


# In[25]:


columns_to_drop = ['X.1', 'X','Total_Sentiment','Total_sentiment1','Positive','Negative','Sentiment_Type']

numerical_column1=numerical_column.drop(columns=columns_to_drop)
numerical_column1


# In[26]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_column1)


# In[27]:


pca = PCA(n_components=9)  # Specify the number of components you want to retain
pca.fit(scaled_data)


# In[28]:


pca_data = pca.transform(scaled_data)


# In[29]:


pca_data


# In[30]:


print("Principal Components:", pca.components_)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)


# In[31]:


df_pca = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2', 'PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9',])


# In[32]:


df1= df["Sentiment_Type"]


# In[33]:


correlation_matrix = df_pca.corr()
correlation_matrix


# In[34]:


plt.figure(figsize=(10,4))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()


# In[35]:


df_pca


# In[36]:


concatenated_df = pd.concat([df_pca,df1 ], axis=1)
concatenated_df


# In[37]:


import pandas as pd

# Sample DataFrame
df = concatenated_df['Sentiment_Type']

# Perform one-hot encoding
one_hot_encoded = pd.get_dummies(df, prefix='Category')

# Concatenate one-hot encoded columns with the original DataFrame
df_encoded = pd.concat([df, one_hot_encoded], axis=1)

# Print the result
print("Original DataFrame:")
print(df)
print("\nOne-hot encoded DataFrame:")
print(df_encoded)


# In[ ]:





# In[38]:


import pandas as pd

# Sample Series (Single column)
s =  concatenated_df['Sentiment_Type']

# Perform one-hot encoding
one_hot_encoded = pd.get_dummies(s, prefix='Category')

# Concatenate one-hot encoded columns with the original Series
s_encoded = pd.concat([s, one_hot_encoded], axis=1)

# Print the result
print("Original Series:")
print(s)
print("\nOne-hot encoded DataFrame:")
print(s_encoded)


# In[39]:


s


# In[40]:


dd=pd.Series(s)
mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
s_mapped = dd.map(mapping)


# In[41]:


s_mapped


# In[42]:


cleaned_data = pd.concat([df_pca,s_mapped], axis=1)


# In[43]:


cleaned_data


# In[44]:


a=[0.28419813, 0.17214201, 0.10619018, 0.10002927, 0.08928461, 0.07789951 ,0.06823332, 0.0482505 , 0.03905827]
sum(a)


# In[45]:


scaled_data


# In[166]:


pca = PCA()
pca.fit(scaled_data)

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained variance by number of PCA components')
plt.grid(True)
plt.savefig('pca.png')
plt.show()


# In[50]:


features = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9']
X = cleaned_data[features]  # Extract features from your DataFrame
y = cleaned_data['Sentiment_Type']# Replace 'target' with the name of your target variable column


# ### Visual Representation of Sentiment Types

# In[51]:


# Distribution of Total Sentiment
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=20, kde=True, color='skyblue')
plt.title('Distribution of Sentiment Type')
plt.xlabel(' Sentiment Type')
plt.ylabel('Frequency')
plt.show()


# # Machine Learning Models

# In[ ]:





#  ### Logistic Regression Fitting   

# In[53]:


cleaned_data.columns


# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

# Step 1: Prepare Data
# Assuming 'X' contains your features and 'y' contains your target variable
features = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9']
X = cleaned_data[features]  # Extract features from your DataFrame
y = cleaned_data['Sentiment_Type']# Replace 'target' with the name of your target variable column


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=62)

# Step 3: Fit Multinomial Logistic Regression
# Set multi_class parameter to 'multinomial' or 'auto' for multinomial logistic regression
logreg = LogisticRegression(multi_class='multinomial', max_iter=100)
logreg.fit(X_train, y_train)

# Step 4: Evaluate Model
# Predict on the test set
y_pred_logis = logreg.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_logis))


# In[49]:


print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_logis))


# In[47]:


y_pred_logis


# ## SVM(Support Vector Machines)

# In[70]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error


# #### kernel is Radial Basis Function(rbf)

# In[75]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Scale the features (optional but recommended for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SVM model
svm_model = SVC(kernel='rbf')  # You can choose different kernels like 'rbf', 'poly', etc.

# Fit the SVM model to the training data
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred_svm)
print("Mean Squared Error:", mse)


# In[76]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))


# In[77]:


# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))


# ## Decision Tree

# In[32]:


from sklearn.tree import DecisionTreeClassifier


# In[33]:


from sklearn.metrics import confusion_matrix, classification_report


# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred_tree=model.predict(X_test)
y_pred_tree


# In[52]:


confusion_matrix(y_pred_tree,y_test)


# In[53]:


print(classification_report(y_pred_tree,y_test))


# In[196]:


model.feature_importances_


# In[54]:


from sklearn.tree import plot_tree


# In[58]:


def report_model(mod):
    mod_pred = mod.predict(X_test)
    
    print('\n')
    print(classification_report(y_test,mod_pred))
    print('\n')
    plt.figure(figsize=(12,6))
    plot_tree(mod, feature_names=list(X.columns));


# In[59]:


mod = DecisionTreeClassifier(max_depth=2)
mod.fit(X_train,y_train)


# In[60]:


report_model(mod)


# In[61]:


max_leaf_tree = DecisionTreeClassifier(max_leaf_nodes=4)
max_leaf_tree.fit(X_train,y_train)


# In[62]:


report_model(max_leaf_tree)


# ### Random Forest 

# In[63]:


from sklearn.ensemble import RandomForestClassifier


# In[64]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=101, max_features="sqrt")
rf_model
rf_model.fit(X_train,y_train)
y_pred_r= rf_model.predict(X_test)
y_pred_r


# In[66]:


confusion_matrix(y_test,y_pred_r)


# In[67]:


print(classification_report(y_test,y_pred_r))


# In[68]:


test_error = []

for n in range(1,100):
    modrf = RandomForestClassifier(n_estimators=n)
    modrf.fit(X_train,y_train)
    test_preds = modrf.predict(X_test)
    test_error.append(1-accuracy_score(test_preds,y_test))


# In[221]:


plt.plot(range(1,100),test_error,label='Test Error')
plt.legend()
plt.grid()
plt.show()


# In[227]:


rf_model = RandomForestClassifier(n_estimators=96, random_state=101, max_features="sqrt")
rf_model
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
confusion_matrix(y_test,y_pred)


# In[228]:


print(classification_report(y_test,y_pred))


# In[229]:


rf_model = RandomForestClassifier(n_estimators=77, random_state=101, max_features="sqrt")
rf_model
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
confusion_matrix(y_test,y_pred)


# In[230]:


print(classification_report(y_test,y_pred))


# In[231]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the range of n_estimators values to search
param_grid = {'n_estimators': [40, 50, 60, 70, 80, 90, 100]}

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameter value
best_n_estimators = grid_search.best_params_['n_estimators']
print("Best n_estimators:", best_n_estimators)


# ## Comparision

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Generate some synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train each model and get predictions
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

# Plotting
plt.figure(figsize=(10, 6))

# Plotting true values for reference
sns.lineplot(x=y_test, y=y_test, label='True Values', color='blue', linestyle='-')

# Plotting predicted values for each model with markers
markers = ['o', 's', '^', 'D']
for i, (name, y_pred) in enumerate(predictions.items()):
    sns.lineplot(x=y_test, y=y_pred, label=name, marker=markers[i % len(markers)])

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Predicted Values vs True Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

