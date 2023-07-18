#Importando librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

#Herramientas de sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#XGBoost
import xgboost as xgb

#extra
import configparser 

import warnings
warnings.filterwarnings("ignore")

class Model():
    def __init__(self, model=None, params=None):
        '''
        'model' corresponde al modelo que ya está entrenado, si a este se le asigna None entonces el código deberá entrenar.
        'params' corresponde a un diccionario con los parametros del modelo para entrenar 
        ''' 

        if model is None:
            self.model = xgb.XGBClassifier(random_state=params['random_state'],
                                      learning_rate=params['learning_rate'],
                                      max_depth=params['max_depth'],
                                      colsample_bytree=params['colsample_bytree'],
                                      subsample=params['subsample'])

        elif model is not None:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model)

    def train(self, x_train, y_train,x_val, y_val):
        '''método para realizar el entrenamiento'''

        self.model.fit(x_train,y_train)
        y_pred = self.model.predict(x_val)
        
        cm = confusion_matrix(y_val,y_pred)
        print(cm)
        print("--"*10)
        print(round(accuracy_score(y_val,y_pred),3)*100,"%")
        print("--"*10)
        print(classification_report(y_val,y_pred))


    def predict(self,x):
        '''método para hacer la predicción'''

        y_pred = self.model.predict(x)

        return y_pred

#Parámetros a modificar
config = configparser.ConfigParser()
config.read('config.ini') 

TRAIN = config.getboolean('params','train')

PARAMS = {'random_state':config.getint('params','random_state'),
          'learning_rate':config.getfloat('params','learning_rate'),
          'max_depth':config.getint('params','max_depth'),
          'colsample_bytree':config.getfloat('params','colsample_bytree'),
          'subsample':config.getfloat('params','subsample')} 

MODELO = config.get('params','model')

#Bloque de preprocesamiento de datos
data = pd.read_csv('final_feat.csv',index_col='Unnamed: 0')
X = data.drop(columns=['atraso_15']) if TRAIN else data
y = data['atraso_15']
X_feat = pd.concat([pd.get_dummies(data['OPERA'], prefix = 'OPERA'),pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), pd.get_dummies(data['MES'], prefix = 'MES')], axis = 1)

#####
if TRAIN:
    xgboost = Model(params=PARAMS)
    #Separando el archivo
    X_train, X_test, y_train, y_test = train_test_split(X_feat,y,test_size=0.2,stratify=y,random_state=23)

    xgboost.train(x_train=X_train,x_val=X_test,y_train=y_train,y_val=y_test)
    xgboost.model.save_model('xgb_trained_model.json')

else:
    ##Prueba:
    xgboost = Model(model=MODELO,params=PARAMS)
    X_feat = np.array(X_feat.iloc[-3:,:])
    pred = xgboost.predict(X_feat)
    pred_file = {'index':X.iloc[-3:,:].index,'atraso_15':pred}
    df = pd.DataFrame(pred_file)
    df.to_csv("predicción.csv")
