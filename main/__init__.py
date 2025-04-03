import pickle
from django.conf import settings

model_path = settings.MODEL_PATH
scaler_path =  settings.SCALER_PATH
caliber_path = settings.CALIBER_PATH

with open(model_path, "rb") as file:
    MODEL = pickle.load(file)

with open(scaler_path, "rb") as file2:
    SCALER = pickle.load(file2)


with open(caliber_path, "rb") as file3:
    CALIBER = pickle.load(file3)



