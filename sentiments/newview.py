#https://www.urdunlp.com/2020/02/urdu-sentiment-classification.html
from django.shortcuts import render
import numpy as np
import pandas as pd
from textblob import TextBlob
from googletrans import Translator
import csv
import pickle
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
import emoji
import re
from drf_yasg.utils import swagger_auto_schema
import fasttext
from .serializers import GetSentimentSerializer,PostSentimentSerializer
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('urdu-corpus')
from django.http import HttpResponse
# nltk.download('omw-1.4')
# nltk.download('urdu')



def test_Roman_trained_model(test_text):
    saved_model_dic =pickle.load(open("static/RomanUrduModel.sav", 'rb'))
    saved_clf = saved_model_dic['model']
    saved_vectorizer = saved_model_dic['vectorizer']
    new_test_vecs = saved_vectorizer.fit_transform(test_text)
    return saved_clf.predict(new_test_vecs)

def test_urdu_trained_model(test_text):
    saved_model_dic =pickle.load(open("static/UrduModel.sav", 'rb'))
    saved_clf = saved_model_dic['model']
    saved_vectorizer = saved_model_dic['vectorizer']
    new_test_vecs = saved_vectorizer.fit_transform(test_text)
    return saved_clf.predict(new_test_vecs)
#1.0.2
class getSentiment(APIView):
    # authentication_classes=[TokenAuthentication]
    # permission_classes = [IsAuthenticated]
    @swagger_auto_schema(request_body=GetSentimentSerializer)
    def post(self, request):
        text=request.data['text']
        print("emoji.demojize(text)",emoji.demojize(text))
        text=emoji.demojize(text)
        text=text.replace("_"," ")
        text=text.replace(":","")
        print("text: ",text)
        model = fasttext.load_model('static/lid.176.ftz')
        lang=model.predict(text)
        lang=lang[0][0]
        print("lang===>",lang)
        if lang=="__label__en":       
            print(lang)
            print("English")
            b = TextBlob(text)
            if b.sentiment.polarity > 0:
                return Response("Positive")
                
            elif b.sentiment.polarity == 0:
                return Response("Neutral")
            else:
                return Response("Negative")
        elif lang=="__label__ur":
            print("Urdu")
            predict_result=test_urdu_trained_model([text])
            sentiment = np.array(predict_result[0])
            print("sentiment api",sentiment)
            if sentiment=="positive":
                return Response("Positive")
            elif sentiment=="neutral":
                return Response("Neutral")
            else:
                return Response("Negative")
        else:
            predict_result=test_Roman_trained_model([text])
            print("Roman Urdu")
            sentiment = np.array(predict_result)
            print("sentiment api",sentiment)
            if sentiment=="Positive":
                return Response("Positive")
            elif sentiment=="Neutral":
                return Response("Neutral")
            else:
                return Response("Negative")

def writer(data, filename):
    with open (filename, "a", newline = "") as csvfile:
        movies = csv.writer(csvfile)
        for x in data:
            movies.writerow(x)

class postSentiment(APIView):
    # authentication_classes=[TokenAuthentication]
    # permission_classes = [IsAuthenticated]
    @swagger_auto_schema(request_body=PostSentimentSerializer)
    def post(self, request):
        Comment =request.data['comment']
        sentiment =request.data['sentiment']
        model = fasttext.load_model('static/lid.176.ftz')
        lang=model.predict(Comment)
        lang=lang[0][0]
        print("lang===>",lang)
        if lang!="__label__en" and lang!="__label__ur":
            filename = "static/OurDataSet.csv"
            data = [ (Comment,sentiment,"")]
            writer(data,filename)
            return Response("successfully Added")
        else:
            return Response("please put Roman Urdu with sentiment")

@csrf_exempt
def createSentiment(request):
    
    if request.method=="POST":
        Comment =request.POST.get('comment')
        sentiment =request.POST.get('sentiment')
        model = fasttext.load_model('static/lid.176.ftz')
        lang=model.predict(Comment)
        lang=lang[0][0]
        print("lang===>",lang)
        if lang!="__label__en" and lang!="__label__ur":
            filename = "static/OurDataSet.csv"
            data = [ (Comment,sentiment,"")]
            writer(data,filename)

    return render(request, 'sentiments/createsentiment.html')
#sentiments-346910
#441039245076

# @csrf_exempt
# def home(request):
#     if request.method=="POST":
#         text =request.POST.get('txt')
#         text=emoji.demojize(text)
#         text=text.replace("_"," ")
#         text=text.replace(":","")
#         model = fasttext.load_model('static/lid.176.ftz')
#         lang=model.predict(text)
#         lang=lang[0][0]
#         print("lang===>",lang)
#         if lang=="__label__en":
#             b = TextBlob(text)
#             if b.sentiment.polarity > 0:
#                 context = {'sent_val':'Positive'}
#                 return render(request, 'sentiments/sentiment.html',context)
#             elif b.sentiment.polarity == 0:
#                 context = {'sent_val':'Neutral'}
#                 return render(request, 'sentiments/sentiment.html',context)
#             else:
#                 print('negative')
#                 context = {'sent_val':'Negative'}
#                 return render(request, 'sentiments/sentiment.html',context)
#         if lang=="__label__ur":
#             predict_result=test_urdu_trained_model([text])
#             sentiment = np.array(predict_result[0])

#             print("sentiment===>",sentiment)
#             context = {
#               'sent_val':sentiment
#             }
#             return render(request, 'sentiments/sentiment.html',context)
            
#         else:
#             predict_result=test_Roman_trained_model([text])
#             sentiment = np.array(predict_result[0])
#             print("sentiment===>",sentiment)
#             context = {
#               'sent_val':sentiment
#             }
#             return render(request, 'sentiments/sentiment.html',context)

    
#     return render(request, 'sentiments/sentiment.html')





@csrf_exempt
def home(request):
    if request.method=="POST":
        text = request.POST.get('txt')
        text = emoji.demojize(text)
        text = text.replace("_"," ")
        text = text.replace(":","")
        text = text.lower()
        text = re.sub(r"\d+", "", text) # Remove numbers
        text = re.sub(r"\s+", " ", text) # Remove extra white spaces
        text = re.sub(r"[^\w\s]", "", text) # Remove special characters
        text = re.sub(r"\b@\b|\b#\b", "", text) # Remove hashtags and mentions
        print("cleaned text: ", text)
        model = fasttext.load_model('static/lid.176.ftz')
        lang=model.predict(text)
        lang=lang[0][0]
        # Tokenization and lemmatization
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        if lang == "__label__ur":
            nltk.download('stopwords')
            urdu_stopwords = ["کے", "کو", "کی", "کا", "ہے", "ہیں", "میں", "اور", "پر"]
            stop_words = set(stopwords.words('urdu') + urdu_stopwords)
            words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        else:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

        text = " ".join(words)
        print("cleaned text: ", text)
        model = fasttext.load_model('static/lid.176.ftz')
        lang = model.predict(text)[0][0]
        print("lang===>",lang)
        if lang == "__label__en":
            print(lang)
            print("English")
            b = TextBlob(text)
            if b.sentiment.polarity > 0:
                context = {'sent_val':'Positive'}
                return render(request, 'sentiments/sentiment.html',context)
            elif b.sentiment.polarity == 0:
                context = {'sent_val':'Neutral'}
                return render(request, 'sentiments/sentiment.html',context)
            else:
                context = {'sent_val':'Negative'}
                return render(request, 'sentiments/sentiment.html',context)
                
        elif lang == "__label__ur":
            print("Urdu")
            predict_result = test_urdu_trained_model([text])
            sentiment = np.array(predict_result[0])
            print("sentiment api",sentiment)
            context = {'sent_val':sentiment}
            return render(request, 'sentiments/sentiment.html',context)
                
        else:
            predict_result = test_Roman_trained_model([text])
            print("Roman Urdu")
            sentiment = np.array(predict_result)
            print("sentiment api",sentiment)
            context = {'sent_val':sentiment}
            return render(request, 'sentiments/sentiment.html',context)

    return render(request, 'sentiments/sentiment.html')



# updated
# class GetSentiment(APIView):
#     # authentication_classes=[TokenAuthentication]
#     # permission_classes = [IsAuthenticated]
#     @swagger_auto_schema(request_body=GetSentimentSerializer)
#     def post(self, request):
#         text = request.data['text']
#         print("emoji.demojize(text)",emoji.demojize(text))
#         text = emoji.demojize(text)
#         text = text.replace("_"," ")
#         text = text.replace(":","")
#         print("text: ",text)

#         # Text cleaning
#         text = text.lower()
#         text = re.sub(r"\d+", "", text) # Remove numbers
#         text = re.sub(r"\s+", " ", text) # Remove extra white spaces
#         text = re.sub(r"[^\w\s]", "", text) # Remove special characters
#         text = re.sub(r"\b@\b|\b#\b", "", text) # Remove hashtags and mentions

#         # Tokenization and lemmatization
#         nltk.download('wordnet')
#         lemmatizer = WordNetLemmatizer()
#         words = nltk.word_tokenize(text)
#         if lang == "__label__ur":
#             nltk.download('stopwords')
#             urdu_stopwords = ["کے", "کو", "کی", "کا", "ہے", "ہیں", "میں", "اور", "پر"]
#             stop_words = set(stopwords.words('urdu') + urdu_stopwords)
#             words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
#         else:
#             nltk.download('stopwords')
#             stop_words = set(stopwords.words('english'))
#             words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

#         text = " ".join(words)
#         print("cleaned text: ", text)

#         model = fasttext.load_model('static/lid.176.ftz')
#         lang = model.predict(text)[0][0]
#         print("lang===>",lang)
        
#         if lang == "__label__en":
#             print(lang)
#             print("English")
#             b = TextBlob(text)
#             if b.sentiment.polarity > 0:
#                 return Response("Positive")
#             elif b.sentiment.polarity == 0:
#                 return Response("Neutral")
#             else:
#                 return Response("Negative")
                
#         elif lang == "__label__ur":
#             print("Urdu")
#             predict_result = test_urdu_trained_model([text])
#             sentiment = np.array(predict_result[0])
#             print("sentiment api",sentiment)
#             if sentiment == "positive":
#                 return Response("Positive")
#             elif sentiment == "neutral":
#                 return Response("Neutral")
#             else:
#                 return Response("Negative")
                
#         else:
#             predict_result = test_Roman_trained_model([text])
#             print("Roman Urdu")
#             sentiment = np.array(predict_result)
#             print("sentiment api",sentiment)
#             if sentiment == "Positive":
#                 return Response("Positive")
#             elif sentiment == "Neutral":
#                 return Response("Neutral")
#             else:
#                 return Response("Negative")






# Define stop words
# english_stopwords = set(stopwords.words('english'))
# urdu_stopwords = set(["کے", "کو", "کی", "کا", "ہے", "ہیں", "میں", "اور", "پر"])
# @csrf_exempt
# def home(request):
#     if request.method=="POST":
#         text =request.POST.get('txt')
#         text=emoji.demojize(text)
#         text=text.replace("_"," ")
#         text=text.replace(":","")
#         model = fasttext.load_model('static/lid.176.ftz')
#         lang=model.predict(text)
#         lang=lang[0][0]
#         print("lang===>",lang)
#         # Text cleaning
#         text = text.lower()
#         text = re.sub(r"\d+", "", text) # Remove numbers
#         text = re.sub(r"\s+", " ", text) # Remove extra white spaces
#         text = re.sub(r"[^\w\s]", "", text) # Remove special characters
#         text = re.sub(r"\b@\b|\b#\b", "", text) # Remove hashtags and mentions
        
#         # Tokenization and lemmatization
#         words = nltk.word_tokenize(text)
#         if lang=="__label__ur":
#             stop_words = urdu_stopwords
#             lemmatizer = UrduLemmatizer()
#             words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
#         else:
#             stop_words = english_stopwords
#             lemmatizer = WordNetLemmatizer()
#             words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        
#         text = " ".join(words)
#         print("cleaned text: ", text)

#         if lang=="__label__en":
#             b = TextBlob(text)
#             if b.sentiment.polarity > 0:
#                 context = {'sent_val':'Positive'}
#                 return render(request, 'sentiments/sentiment.html',context)
#             elif b.sentiment.polarity == 0:
#                 context = {'sent_val':'Neutral'}
#                 return render(request, 'sentiments/sentiment.html',context)
#             else:
#                 print('negative')
#                 context = {'sent_val':'Negative'}
#                 return render(request, 'sentiments/sentiment.html',context)
#         if lang=="__label__ur":
#             predict_result=test_urdu_trained_model([text])
#             sentiment = np.array(predict_result[0])
#             print("sentiment===>",sentiment)
#             context = {
#               'sent_val':sentiment
#             }
#             return render(request, 'sentiments/sentiment.html',context)
#         else:
#             predict_result=test_Roman_trained_model([text])
#             sentiment = np.array(predict_result[0])
#             print("sentiment===>",sentiment)
#             context = {
#               'sent_val':sentiment
#             }
#             return render(request, 'sentiments/sentiment.html',context)
#     return render(request, 'sentiments/sentiment.html')
