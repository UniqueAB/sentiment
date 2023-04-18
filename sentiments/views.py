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
#from django.contrib.auth.decorators import login_required


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


class getSentiment(APIView):
    authentication_classes=[TokenAuthentication]
    permission_classes = [IsAuthenticated]
    @swagger_auto_schema(request_body=GetSentimentSerializer)
    def post(self, request):
        text=request.data['text']
        text=emoji.demojize(text)
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
    authentication_classes=[TokenAuthentication]
    permission_classes = [IsAuthenticated]
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

@csrf_exempt
def home(request):
    if request.method=="POST":
        text =request.POST.get('txt')
        text=emoji.demojize(text)
        text=text.replace("_"," ")
        text=text.replace(":","")
        text = text.lower()
        text = re.sub(r"\d+", "", text) # Remove numbers
        text = re.sub(r"\s+", " ", text) # Remove extra white spaces
        text = re.sub(r"[^\w\s]", "", text) # Remove special characters
        text = re.sub(r"\b@\b|\b#\b", "", text) # Remove hashtags and mentions
        print("cleaned text: ", text)
        model = fasttext.load_model('static/lid.176.ftz')
        lang=model.predict(text)
        lang=lang[0][0]
        print("lang===>",lang)
        if lang=="__label__en":
            b = TextBlob(text)
            if b.sentiment.polarity > 0:
                context = {'sent_val':'Positive'}
                return render(request, 'sentiments/sentiment.html',context)
            elif b.sentiment.polarity == 0:
                context = {'sent_val':'Neutral'}
                return render(request, 'sentiments/sentiment.html',context)
            else:
                print('negative')
                context = {'sent_val':'Negative'}
                return render(request, 'sentiments/sentiment.html',context)
        if lang=="__label__ur":
            predict_result=test_urdu_trained_model([text])
            sentiment = np.array(predict_result[0])

            print("sentiment===>",sentiment)
            context = {
              'sent_val':sentiment
            }
            return render(request, 'sentiments/sentiment.html',context)
        else:
            predict_result=test_Roman_trained_model([text])
            sentiment = np.array(predict_result[0])
            print("sentiment===>",sentiment)
            context = {
              'sent_val':sentiment
            }
            return render(request, 'sentiments/sentiment.html',context)
    return render(request, 'sentiments/sentiment.html')

