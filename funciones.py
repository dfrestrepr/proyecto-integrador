# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:20:05 2019

@author: Pablo Saldarriaga
"""
### Paquetes necesarios para el funcionamiento de las funciones en este script
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import distance
#%%
logger = logging.getLogger(__name__)
#%%

### La funcion toma como entrada el conjunto de datos en el cual se desea
### realizar la deteccion de outliers multivariante. El metodo utiliza la
### distancia de Mahalanobis y la matriz de covarianza habitual. Se define
### alpha como el percentil en el cual se realizara el corte de la matriz de 
### distancias para determinar que registros son considerados outliers. 
def outlier_detection_mahal(X,alpha = 0.95):
    logger.info('Comienza la deteccion de outliers')
    if len(X)==0:
        logger.info('No hay datos para realizar deteccion de outliers')
        return None
    try:
        X = np.array(X)
        X_arr = np.array(X)
        X_mean = X_arr.mean(axis = 0)
        cov = np.array(pd.DataFrame(X).cov())
        inv_cov = np.linalg.inv(cov)
        
        dist = []
        i=0
        while i<len(X):
            distances = distance.mahalanobis(X_arr[i], X_mean, inv_cov)
            dist.append(distances)
            i +=1
        
        dist = np.array(dist)
        cutoff = np.quantile(dist,alpha)
        outliers = (dist>cutoff).astype(int)
    except Exception as e:
        logger.info(f'Error en la deteccion de outliers:{e}')
    
    logger.info('Deteccion de outliers exitoso')
    return outliers


### Esta funcion toma como entrada un dataframe con el conjunto de datos, la
### etiqueta de 1s y 0s, y realiza un stepwise en la regresion logistica para
### conservar las variables mas relevantes para explicar la etiqueta. La
### funcion devuelve una lista con las variables relevantes, y el modelo final
### de la regresion logistica
def stepwise_logistic(X,Y,alpha =0.1):
    
    try:
        n = len(X.columns)
        for i in range(n):
            model = sm.Logit(Y, X).fit()
            pvalues = pd.DataFrame(model.pvalues,columns = ['pvalues']).reset_index()
            if pvalues['pvalues'].max()<alpha:
                break
            to_drop = pvalues[pvalues['pvalues']==pvalues['pvalues'].max()]['index'].values[0]
            X = X.drop(columns = to_drop)
    
        model = sm.Logit(Y, X).fit()
        variables_relevantes = list(X.columns) 
    except Exception as e:
        logger.info(f'Error en el stepwise logistic:{e}')
        model = None
        variables_relevantes = []
        
    return variables_relevantes,model
