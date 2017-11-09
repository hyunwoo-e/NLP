from rest_framework import status
from rest_framework import serializers
from rest_framework.views import APIView
from rest_framework.response import Response
from generation_service.models import Dialog
from generation_service.generator import generate_service

class DialogSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dialog
        fields = ['text']
        # fields = '__all__'

class GenerationAPI(APIView):
    queryset = Dialog.text
    serializer_class = DialogSerializer

    def post(self, request, format=None):
        serializer = DialogSerializer(data=request.data)
        if serializer.is_valid():
            return Response(generate_service(serializer.data), status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)