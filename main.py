import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from flask import Flask
from flask_cors import CORS

# Nombre dataset usado y modelo preentrenado
MODEL_FILENAME = './xgb_trained_model.json'
PARAMS_FILENAME = './final_feat.csv'


# parametros de `config.ini` pero hardcodeado
PARAMS = {
  'train':False,
  'random_state': 93,
  'learning_rate': 0.001,
  'max_depth': 10,
  'colsample_bytree': 0.8,
  'subsample': 1.0
}

class Model():
  
  def __init__(self):
    if PARAMS['train']:
      self.model = xgb.XGBClassifier(
          random_state=PARAMS['random_state'],
          learning_rate=PARAMS['learning_rate'],
          max_depth=PARAMS['max_depth'],
          colsample_bytree=PARAMS['colsample_bytree'],
          subsample=PARAMS['subsample']
          )
    else:
      self.model = xgb.XGBClassifier()
      self.model.load_model(MODEL_FILENAME)
  
  def train(self, x_train, y_train,x_val, y_val):
    '''método para realizar el entrenamiento'''

    self.model.fit(x_train,y_train)
    y_pred = self.model.predict(x_val)
        
    cm = confusion_matrix(y_val,y_pred)
    result = {
      'CR': classification_report(y_val,y_pred),
      'accuracy': round(accuracy_score(y_val,y_pred),3)*100
    }
    #print(cm)
    #print("--"*10)
    #print(round(accuracy_score(y_val,y_pred),3)*100,"%")
    #print("--"*10)
    #print(classification_report(y_val,y_pred))
    return result

  def predict(self, x):
    return self.model.predict(x)

app = Flask(__name__)

CORS(app)

@app.route("/")
def index():
  data = pd.read_csv(PARAMS_FILENAME, index_col='Unnamed: 0')
  X = data.drop(columns=['atraso_15'])
  y = data['atraso_15']
  X_feat = pd.concat([
    pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
    pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
    pd.get_dummies(data['MES'], prefix = 'MES')
  ], axis = 1)

  if PARAMS['train']:
    X_train, X_test, y_train, y_test = train_test_split(X_feat.iloc[:-3,:],y[:-3],test_size=0.2,stratify=y[:-3],random_state=23)
    xgboost = Model()
    result = xgboost.train(x_train=X_train,x_val=X_test,y_train=y_train,y_val=y_test)
    xgboost.model.save_model('xgb_trained_model.json')
    
    return result
  
  else:
    xgboost = Model()
    prediction = xgboost.predict(np.array(X_feat.iloc[-3:,:]))
    pred_file = {
      'ID': X.iloc[-3:,:].index,
      'TIPOVUELO':X['TIPOVUELO'][-3:],
      'OPERA':X['OPERA'][-3:],
      'SIGLADES':X['SIGLADES'][-3:],
      'DIANOM':X['DIANOM'][-3:],
      'atraso_15': prediction,
      }
    df = pd.DataFrame(pred_file)

    return df.to_html()

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
