# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:35:08 2021

@author: LENOVO
"""

import pandas as pd


data = pd.read_csv('phpgNaXZe.csv')
data.head()

#Colocar nombres a las columnas
columnas = ['sbp','Tabaco','ldl','Adiposity','Familia','Tipo','Obesidad','Alcohol','Edad','chd']
data.columns=columnas
data.head()

#Conocer el formato de los datos
data.dtypes

#Conocer los datos nulos
data.isnull().sum()
#Cambiar los datos de Familia y CHD en digitales
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Familia']=encoder.fit_transform(data['Familia'])
data['chd']=encoder.fit_transform(data['chd'])
data.head()

#Escalamos los valores de la columna sbp
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range =(0,100))
data['sbp'] = scale.fit_transform(data['sbp'].values.reshape(-1,1))
data.head()

#Visualizar la obesidad de acuerdo a la edad
data.plot(x='Edad',y='Obesidad',kind='scatter',figsize =(10,5))
#Visualizar el consumo de tabaco de acuerdo a la edad
data.plot(x='Edad',y='Tabaco',kind='scatter',figsize =(10,5))
#Visualizar el consumo de alcohol de acuerdo a la edad
data.plot(x='Edad',y='Alcohol',kind='scatter',figsize =(10,5))
### ANÁLISIS DE MACHINE LEARNING ###
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
#Definir las variable dependiente e independientes
y = data['chd']
X = data.drop('chd', axis =1)
#Separar los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)
#Definir el algoritmo
algoritmo = svm.SVC(kernel ='linear')
#Entrenar el algoritmo
algoritmo.fit(X_train, y_train)
#Realizar una predicción
y_test_pred = algoritmo.predict(X_test)
#Se calcula la matriz de confusión
print(confusion_matrix(y_test, y_test_pred))
#Se calcula la exactitud y precisión del modelo
accuracy_score(y_test, y_test_pred)
precision_score(y_test, y_test_pred)