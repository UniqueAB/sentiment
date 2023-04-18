import numpy as np 
import pandas as pd 
import os
import psutil
import time

# Print initial RAM usage
process = psutil.Process(os.getpid())
print('Initial RAM usage: {:.2f} GB'.format(process.memory_info().rss / 2**30))

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Import Plotting Libararies
import seaborn as sns
import matplotlib.pyplot as plt

# Import Data Preprocessing Libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Machine Learning Models
from sklearn import svm  
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

# Model Evaluation Libraries
from sklearn.metrics import classification_report, confusion_matrix

# Upload file for read and train here
data =  pd.read_csv('data/train.csv')
#test =   pd.read_csv('/content/imdb_urdu_reviews_test.csv')
print('Shape of Training Set: ' , data.shape,'\nShape of Testing Set: ', data.shape)

#data =  pd.concat([train, test]).reset_index(drop=True)
#print(data.shape)

df =  data.copy()

df.head()

sns.countplot( x = 'sentiment', data = df )

le = LabelEncoder()
le.fit(df['sentiment'])
df['encoded_sentiments'] = le.transform(df['sentiment'])

df.head()

X_train, X_test, Y_train, Y_test = train_test_split(df['review'], df['encoded_sentiments'], test_size = 0.20, random_state = 1000)

print('Shape of X_train:', X_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of Y_train:', Y_train.shape)
print('Shape of Y_test:', Y_test.shape)

from sklearn.metrics import accuracy_score
import pickle
max_feature_num = 50000
vectorizer = TfidfVectorizer(max_features=max_feature_num)
train_vecs = vectorizer.fit_transform(X_train)
test_vecs = TfidfVectorizer(max_features=max_feature_num, vocabulary=vectorizer.vocabulary_).fit_transform(X_test)
SVM = svm.LinearSVC(max_iter=100)
model=SVM.fit(train_vecs, Y_train)

# Print current RAM usage and CPU utilization
process = psutil.Process(os.getpid())
print('Current RAM usage: {:.2f} GB'.format(process.memory_info().rss / 2**30))
print('Current CPU utilization: {}%'.format(psutil.cpu_percent()))

test_predictionSVM = SVM.predict(test_vecs)
acc = accuracy_score(Y_test, test_predictionSVM)
print('Accuracy is {} '.format(acc))

# Output model saved
filename='saved_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Print final RAM usage and CPU utilization
process = psutil.Process(os.getpid())
print('Final RAM usage: {:.2f} GB'.format(process.memory_info().rss / 2**30))
print('Final CPU utilization: {}%'.format(psutil.cpu_percent()))
