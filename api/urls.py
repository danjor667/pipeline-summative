from django.urls import path

from api.views import UploadView

urlpatterns = [
    path('upload/', UploadView.as_view(), name='uploadapi'),
]