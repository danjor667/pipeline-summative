import requests


def upload_data():
    url = "http://127.0.0.1:8000/api/upload/"

    file_path = "diabetes.csv"

    files = {"dataset": open(file_path, "rb")}

    response = requests.post(url, files=files)

    return response.status_code


if __name__ == "__main__":
    print(upload_data())