from django.urls import path
from .import views
urlpatterns = [
   # path('',views.home,name="home"),
   # path('createsentiment/',views.createSentiment,name="createsentiment"),
    path('getsentiment',views.getSentiment.as_view()),
    path('postsentiment',views.postSentiment.as_view()),


]
