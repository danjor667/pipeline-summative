import io

from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from main.tasks import preprocess

# Create your views here.


class UploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        file = request.FILES.get('dataset')
        file_data = file.read()

        task  = preprocess.delay(file_data, file.name)
        return JsonResponse({'task_id': task.id}, status=status.HTTP_200_OK)


    def get(self, request):
        return JsonResponse({'task_id': None}, status=status.HTTP_200_OK)

