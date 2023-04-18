import numpy as np 
import pandas as pd 
import os
import psutil
import time
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm  
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# Measure CPU utilization and elapsed time
process = psutil.Process(os.getpid())
start = time.time()

# Load data
data = pd.read_csv('data/train.csv')

# Encode target variable
le = LabelEncoder()
le.fit(data['sentiment'])
data['encoded_sentiments'] = le.transform(data['sentiment'])

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data['review'], data['encoded_sentiments'], test_size = 0.20, random_state = 1000)

# Vectorize data using TF-IDF
max_feature_num = 50000
vectorizer = TfidfVectorizer(max_features=max_feature_num)
train_vecs = vectorizer.fit_transform(X_train)
test_vecs = TfidfVectorizer(max_features=max_feature_num, vocabulary=vectorizer.vocabulary_).fit_transform(X_test)

# Train SVM model and predict on test set
SVM = svm.LinearSVC(max_iter=100)

# Print real-time CPU and RAM utilization during training process
for i in range(5):
    print(f"CPU utilization: {process.cpu_percent()}%")
    print(f"RAM utilization: {psutil.virtual_memory().percent}%")
    print("\n")
    model = SVM.fit(train_vecs, Y_train)
    test_predictionSVM = SVM.predict(test_vecs)
    acc = accuracy_score(Y_test, test_predictionSVM)
    print(f"Accuracy is {acc}")
    time.sleep(1)

# Save model
filename='saved_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Print CPU utilization and elapsed time
print(f"CPU utilization: {process.cpu_percent()}%")
print(f"Elapsed time: {time.time() - start} seconds")

# Plot distribution of target variable
sns.countplot(x='sentiment', data=data)
plt.show()
