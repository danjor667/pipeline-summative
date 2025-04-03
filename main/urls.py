from django.urls import path
from .views import index, login_view, predict, about, signup, prediction_result, logout_view, \
    process_upload_view

urlpatterns = [
    path("", index, name="index"),
    path("login/", login_view, name="login"),
    path("predict/", predict, name="predict"),
    path("upload/", process_upload_view, name="upload"),
    path('about/', about, name="about"),
    path("signup/", signup, name="signup"),
    path("prediction-result/", prediction_result, name="prediction_result"),
    path("logout/", logout_view, name="logout"),
]
