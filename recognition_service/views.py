from rest_framework import status
from rest_framework import serializers
from rest_framework.views import APIView
from rest_framework.response import Response
from recognition_service.models import Dialog
from recognition_service.recognizer import recognize_service
from recognition_service.recognizer import tokenize

class DialogSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dialog
        fields = ['text']
        # fields = '__all__'

class RecognitionAPI(APIView):
    queryset = Dialog.text
    serializer_class = DialogSerializer

    def post(self, request, format=None):
        serializer = DialogSerializer(data=request.data)
        if serializer.is_valid():
            #return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(recognize_service(serializer.data), status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)