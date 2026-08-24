from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import string
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import os
import mlflow
from dotenv import load_dotenv

import pandas as pd
load_dotenv()

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "shanshad0999"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "shanshad0999"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')




app = FastAPI()

def load_model(model_name):
    try:
        model=mlflow.pyfunc.load_model(f"models:/{model_name}@production")
        return model
    except Exception as e:
        return e


model = load_model("my_model")
templates = Jinja2Templates(directory="backend/templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request}
    )


@app.post("/predict")
def predict(request: Request, text: str = Form(...)):
    text = normalize_text(text)

    features = vectorizer.transform([text])

    features_df = pd.DataFrame(features.toarray(),
        columns=[str(i) for i in range(features.shape[1])])

    prediction = model.predict(features_df)

    sentiment = label_encoder.inverse_transform(prediction)[0]

    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request, "text": sentiment}
    )