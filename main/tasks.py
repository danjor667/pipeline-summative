import io
import os
import pickle
from datetime import datetime

from celery import shared_task
import pandas as pd
from sklearn.model_selection import train_test_split
from main import MODEL, SCALER
from django.conf import settings

@shared_task
def preprocess(file_data, file_name):
    ## write the file_data in a csv file before trying to open it with pd.read_csv()

    upload_dir = os.path.join(settings.MEDIA_ROOT, "training_data")
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(upload_dir, file_name)

    with open(file_path, 'wb') as f:
        f.write(file_data)

    data = pd.read_csv(file_path)

    features_names = ["Pregnancies",
                      "Glucose",
                      "BloodPressure",
                      "SkinThickness",
                      "Insulin",
                      "BMI",
                      "DiabetesPedigreeFunction",
                      "Age",
                      "Outcome"
                      ]

    # assert(len(data.columns) == 9)
    # assert(all([name in data.columns for name in features_names]))

    if len(data.columns) != len(features_names):
        return(
            {"msg": "please provide a valide csv"}
        )

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X = SCALER.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Fine-tune the model
    MODEL.fit(X_train, y_train)

    with open(settings.MODEL_PATH, 'wb') as file:
        pickle.dump(MODEL, file)

    return(
        {"msg": "Model trained successfully",  "timestamp": datetime.now(), "accuracy": MODEL.score(X_test, y_test)}
    )
