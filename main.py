from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi import Form
import os
import csv
import requests
import json
import uvicorn
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_transformers import RobertaTokenizer, RobertaModel
import joblib
import pandas as pd
import logging
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

used_labels = {0: "Admiration", 2: "Anger", 3: "Annoyance", 8: "Desire", 9: "Disappointment", 10: "Disapproval", 11: "Disgust", 14: "Fear", 18: "Love", 20: "Optimism", 22: "Realisation", 24: "Remorse", 26: "Surprise", 27: "Neutral"}

# Load your model here - replace with your model and vectorizer paths
model = joblib.load('model.joblib')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model.to(device)

# Takes each sentence from the dataset and converts it into tokens whilst adding in the required special tokens
def tokenize(text, max_length = 50, cls_token = True, sep_token = True):

  tokens = tokenizer.tokenize(text)

  if len(tokens) > max_length:
        tokens = tokens[0:(max_length)]
  
  if cls_token:
      tokens.insert(0, tokenizer.cls_token)

  if sep_token:
      tokens.append(tokenizer.sep_token)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)

  return torch.tensor(input_ids).unsqueeze(0), input_mask


class Item(BaseModel):
    text: str


app = FastAPI(

    title="COM3029 Coursework 2",
    description= "testing if this works ",
    version="0.1",
)




@app.get("/home")
def home():
    return ("Home page")

@app.get("/")
def read_root():
    return FileResponse('index.html')


@app.post("/predict")
async def predict(text: str = Form(...)):
    inputs = tokenize(text)[0].to(device)
    
    outputs = model.forward(inputs)[0]
    _, prediction = torch.max(outputs.data, 1)
    print(prediction)

    # Create a DataFrame for this interaction
    df = pd.DataFrame({
        'Date and Time': [datetime.now()],
        'Input': [text],
        'Emotion Predicted': [used_labels.get(list(used_labels.keys())[prediction.item()])]
    }, columns=['Date and Time', 'Input', 'Emotion Predicted'])
    
   
    if not os.path.isfile('interaction_log.csv'):
        #print(df)
        df.to_csv('interaction_log.csv', header = True, index=False)
    else: # else it exists so append without writing the header
       # print(df)
        df.to_csv('interaction_log.csv', mode='a', header=False, index=False)


    logging.info('Input: %s, Prediction: %s', text, used_labels.get(list(used_labels.keys())[prediction.item()]))
    #return {"The predicted emotion is": used_labels.get(list(used_labels.keys())[prediction.item()])}
    return f"The predicted emotion is {used_labels.get(list(used_labels.keys())[prediction.item()])}"


if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0', port=8000)



