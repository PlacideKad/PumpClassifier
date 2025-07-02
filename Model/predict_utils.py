import torch
import pandas as pd
import joblib
from Model.model_architecture import PumpClassifier

scaler=joblib.load('Model/scaler.save')
le=joblib.load('Model/label_encoder.save')
model=PumpClassifier()
model.load_state_dict(torch.load('Model/pumpPredictionModel.pt',weights_only=True))

model.eval()
def pumpModelPrediction(dataArray):
  """
  Cette fontion prend un vecteur ligne de 4 elements en parametre:
  - le premier element correspond à la hauteur manométrique en mètres
  - le deuxième element correspond au débit en fonctionnement de la pompe en litres par minutes
  - le troisième correspond au débit nominal de la pompe en mètres cube par heure
  - le quatrième  correspond au diamètre de la pompe en pouces.

  Le modèle actuel n'est entrainé que sur des donnés de pompes de 8 pouces de diamètre ayant un débit nominal
  de 100 ou 130 m^3/h.
  """
  df_input=pd.DataFrame([[dataArray[0],dataArray[1]]],columns=['col1','col2'])
  scaled=scaler.transform(df_input)[0]
  col3=1 if dataArray[2]==130 else 0
  col4=dataArray[3]
  features=[scaled[0],scaled[1],col3,col4]
  input_tensor=torch.tensor([features],dtype=torch.float32)

  model.eval()
  with torch.no_grad():
    output=model(input_tensor)
    pred_index=torch.argmax(output,dim=1).item()
    return le.inverse_transform([pred_index])[0]