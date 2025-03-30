import time

from django.contrib import messages
from django.contrib.auth import authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, redirect

from .tasks import preprocess
import pandas as pd
from django.contrib.auth import login
from main import SCALER, MODEL

# Create your views here.


def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']
        user = User.objects.create_user(username, email, password)
        user.is_active = True
        user.save()
        return render(request, "main/login.html")
    return  render(request, "main/signup.html", context={})



def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('predict')
        else:
            return render(request, 'main/login.html', context={'error': 'Invalid credentials'})
    return render(request, 'main/login.html', context={})

def logout_view(request):
    logout(request)
    return redirect('index')

def about(request):
    return render(request, 'main/about.html', context={})


def predict(request):
    global SCALER
    global MODEL

    if request.method == 'POST':
        age = request.POST.get('age')
        pregnancies = request.POST.get('pregnancies')
        bmi = request.POST.get('bmi')
        glucose = request.POST.get('glucose')
        skin_thickness = request.POST.get('skinThickness')
        insulin = request.POST.get('insulin')
        blood_pressure = request.POST.get('diastolic')
        dpf = request.POST.get('dpf')
        X = [[age, pregnancies, bmi, glucose, skin_thickness, insulin, blood_pressure, dpf]]
        features_names = ["Pregnancies",
                          "Glucose",
                          "BloodPressure",
                          "SkinThickness",
                          "Insulin",
                          "BMI",
                          "DiabetesPedigreeFunction",
                          "Age"
                          ]

        df = pd.DataFrame(X, columns=features_names)


        scaled_df = SCALER.fit_transform(df)
        prediction = MODEL.predict(scaled_df)
        probabilities = MODEL.predict_proba(scaled_df)


        return render(request, "main/prediction_result.html", context={"prediction": prediction, "probabilities": probabilities[0][1] * 100})

    return render(request, 'main/predict.html', context={})




def index(request):
    return render(request, 'main/index.html', context={})

def prediction_result(request):
    return render(request, 'main/prediction_result.html', context={})


@login_required
def process_upload(request):
    if request.method == 'POST':

        dataset = request.FILES.get('csv_file')
        dataset_data = dataset.read()

        task = preprocess.delay(dataset_data, dataset.name)
        messages.info(request, 'Dataset uploaded successfully processing has started')
        time.sleep(2)
        result = task.result
        return render(request, "main/upload_result.html", context={"result": result})
    return render(request, 'main/upload2.html', context={})
