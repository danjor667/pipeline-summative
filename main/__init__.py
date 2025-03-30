import pickle
from django.conf import settings

model_path = settings.MODEL_PATH
scaler_path =  settings.SCALER_PATH

with open(model_path, "rb") as file:
    MODEL = pickle.load(file)

with open(scaler_path, "rb") as file2:
    SCALER = pickle.load(file2)

