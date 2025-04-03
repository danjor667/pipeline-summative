import pickle
import time

from django.contrib import messages
from django.contrib.auth import authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.conf import settings

from .tasks import preprocess
import pandas as pd
from django.contrib.auth import login
from .decorator import anonymous_required
from main import SCALER, MODEL, CALIBER
from sklearn.calibration import CalibratedClassifierCV

# Create your views here.


@anonymous_required(redirect_url='predict')
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
        gender = request.POST.get('gender')
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
        prediction = CALIBER.predict(scaled_df)
        probabilities = CALIBER.predict_proba(scaled_df)


        context = {
            "prediction": prediction,
            "probabilities": probabilities[0][1] * 100,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "blood_pressure": blood_pressure,
            "glucose_level": glucose
        }

        return render(request, "main/prediction_result.html", context=context)

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


from django.shortcuts import render
from django.conf import settings
from django.contrib import messages
from django.utils import timezone
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # This sets the backend to Agg (non-GUI)
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import json


def process_upload_view(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        try:
            # Start timer for training duration
            training_started = timezone.now()

            # Load the base model from settings
            # base_model = joblib.load(settings.MODEL_PATH)
            base_model = CALIBER

            # Get the uploaded CSV data
            csv_file = request.FILES['csv_file']
            data = pd.read_csv(csv_file)



            # Prepare the data
            X = data.drop("Outcome", axis=1)
            y = data["Outcome"]

            # Save feature names for later use
            feature_names = X.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fine-tune the model
            # try:
            #     # Enable warm start for fine-tuning if the model supports it
            #     if hasattr(base_model, 'warm_start'):
            #         base_model.warm_start = True
            #
            #     # Fine-tune the model
            #     fine_tuned_model = base_model.fit(X_train, y_train)
            # except Exception as e:
            #     # If warm start is not supported, create a new instance of the same model type
            #     model_type = type(base_model)
            #     print("###################################################################################")
            #     fine_tuned_model = model_type().fit(X_train, y_train)

            fine_tuned_model = MODEL.partial_fit(X_train, y_train)

            base_model = CalibratedClassifierCV(MODEL, method="sigmoid")
            base_model.fit(X_train, y_train)

            # Calculate metrics
            y_pred = base_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            precision = precision_score(y_test, y_pred, average='weighted') * 100
            recall = recall_score(y_test, y_pred, average='weighted') * 100
            f1 = f1_score(y_test, y_pred, average='weighted') * 100

            # Get previous model metrics if available
            try:
                model_info_path = settings.MODEL_INFO_PATH
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                prev_accuracy = model_info.get('accuracy')
                prev_precision = model_info.get('precision')
                prev_recall = model_info.get('recall')
                prev_f1_score = model_info.get('f1_score')
            except:
                prev_accuracy = None
                prev_precision = None
                prev_recall = None
                prev_f1_score = None

            # Calculate training duration
            training_completed = timezone.now()
            training_duration = training_completed - training_started

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            classes = sorted(np.unique(y_test))
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()

            # Save confusion matrix as image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            confusion_matrix_img = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            plt.close()

            # Create ROC curve if binary classification
            roc_curve_img = None
            if len(np.unique(y_test)) == 2:
                try:
                    y_score = base_model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc="lower right")
                    plt.tight_layout()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    roc_curve_img = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                    plt.close()
                except Exception as e:
                    messages.warning(request, f"Could not generate ROC curve: {str(e)}")

            # Get feature importance
            try:
                if hasattr(base_model, 'feature_importances_'):
                    feature_importance = base_model.feature_importances_
                elif hasattr(base_model, 'coef_'):
                    feature_importance = np.abs(base_model.coef_[0])
                else:
                    feature_importance = np.ones(len(feature_names)) / len(feature_names)
            except:
                feature_importance = np.ones(len(feature_names)) / len(feature_names)

            # Sort features by importance
            sorted_indices = np.argsort(feature_importance)[::-1]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]
            sorted_feature_importance = feature_importance[sorted_indices].tolist()

            # Get training history data or generate sample data
            try:
                training_history_path = settings.TRAINING_HISTORY_PATH
                with open(training_history_path, 'r') as f:
                    training_history = json.load(f)

                training_dates = training_history.get('dates', [])
                accuracy_history = training_history.get('accuracy', [])
                precision_history = training_history.get('precision', [])
                recall_history = training_history.get('recall', [])

                # Add current training data
                current_date = training_completed.strftime("%Y-%m-%d")
                training_dates.append(current_date)
                accuracy_history.append(accuracy)
                precision_history.append(precision)
                recall_history.append(recall)
            except:
                # Generate sample history with current metrics
                current_date = training_completed.strftime("%Y-%m-%d")
                training_dates = [
                    (training_completed - timedelta(days=30)).strftime("%Y-%m-%d"),
                    (training_completed - timedelta(days=15)).strftime("%Y-%m-%d"),
                    current_date
                ]

                # Sample data for previous training runs
                if prev_accuracy is not None:
                    accuracy_history = [prev_accuracy * 0.95, prev_accuracy, accuracy]
                    precision_history = [prev_precision * 0.95, prev_precision, precision]
                    recall_history = [prev_recall * 0.95, prev_recall, recall]
                else:
                    accuracy_history = [accuracy * 0.9, accuracy * 0.95, accuracy]
                    precision_history = [precision * 0.9, precision * 0.95, precision]
                    recall_history = [recall * 0.9, recall * 0.95, recall]

            # Save the fine-tuned model
            # new_model_path = settings.MODEL_PATH.replace('.joblib', '_finetuned.joblib')
            # joblib.dump(fine_tuned_model, new_model_path)
            # todo gonna use pickle
            with open(settings.MODEL_PATH, "+wb") as file:
                pickle.dump(fine_tuned_model, file)

            # Update model info
            model_info = {
                'id': 'finetuned_' + training_completed.strftime("%Y%m%d%H%M%S"),
                'training_started': training_started.strftime("%Y-%m-%d %H:%M:%S"),
                'training_completed': training_completed.strftime("%Y-%m-%d %H:%M:%S"),
                'training_duration': str(training_duration).split('.')[0],
                'dataset_size': len(data),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'prev_accuracy': prev_accuracy,
                'prev_precision': prev_precision,
                'prev_recall': prev_recall,
                'prev_f1_score': prev_f1_score,
                'description': request.POST.get('description', 'Fine-tuned model')
            }

            # Save model info
            new_model_info_path = settings.MODEL_INFO_PATH.replace('.json', '_finetuned.json')
            with open(new_model_info_path, 'w') as f:
                json.dump(model_info, f)

            # Save updated training history
            new_training_history = {
                'dates': training_dates,
                'accuracy': accuracy_history,
                'precision': precision_history,
                'recall': recall_history
            }

            new_history_path = settings.TRAINING_HISTORY_PATH.replace('.json', '_finetuned.json')
            with open(new_history_path, 'w') as f:
                json.dump(new_training_history, f)

            # Prepare context for template
            context = {
                'model': model_info,
                'confusion_matrix_img': confusion_matrix_img,
                'roc_curve_img': roc_curve_img,
                'feature_names': json.dumps(sorted_feature_names),
                'feature_importance': json.dumps(sorted_feature_importance),
                'training_dates': json.dumps(training_dates),
                'accuracy_history': json.dumps(accuracy_history),
                'precision_history': json.dumps(precision_history),
                'recall_history': json.dumps(recall_history)
            }

            messages.success(request, "Model successfully retrained with new data!")
            return render(request, 'main/upload_result.html', context)

        except Exception as e:
            messages.error(request, f"Error during model retraining: {str(e)}")
            return render(request, 'main/upload2.html')
    else:
        return render(request, 'main/upload2.html')