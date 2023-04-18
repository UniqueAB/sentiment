from re import S
from rest_framework import serializers


class GetSentimentSerializer(serializers.Serializer):
    text=serializers.CharField()
    
class PostSentimentSerializer(serializers.Serializer):
    comment=serializers.CharField()
    sentiment=serializers.CharField()